from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from chroma.research_baselines import CLASSICAL_METHODS, classical_reconstruction
from chroma.research_benchmark import run_benchmark
from chroma.research_codecs import (
    ffmpeg_capabilities,
    jpeg_roundtrip,
    video_roundtrip,
)
from chroma.research_data import (
    DegradationSpec,
    build_manifest,
    read_rgb,
    rgb_to_ycrcb,
    simulate_420,
    verify_manifest,
    ycrcb_to_rgb,
)
from chroma.research_metrics import (
    delta_e_ciede2000_lab,
    reconstruction_metrics,
)
from chroma.research_models import (
    AblationConfig,
    build_ablation_model,
    chroma_loss,
    load_ablation_checkpoint,
    save_ablation_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_pattern(path: Path, offset: int, size: int = 48) -> None:
    yy, xx = np.mgrid[:size, :size]
    red = (xx * 7 + offset * 13) % 256
    green = (yy * 11 + offset * 17) % 256
    blue = ((xx > yy) * 180 + (xx + yy) * 3 + offset * 19) % 256
    image = np.stack((red, green, blue), axis=2).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


class ManifestTests(unittest.TestCase):
    def test_manifest_is_hash_verified_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_pattern(root / "train" / "a.png", 1)
            write_pattern(root / "test" / "b.png", 2)
            manifest = root / "manifest.jsonl"
            records = build_manifest(
                {"train": root / "train", "test": root / "test"},
                manifest,
                dataset_root=root,
            )
            verification = verify_manifest(records, root)
            self.assertTrue(verification["ok"])
            self.assertEqual(len(records), 2)
            self.assertTrue(manifest.with_suffix(".jsonl.meta.json").is_file())

    def test_manifest_rejects_cross_split_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_pattern(root / "train" / "a.png", 1)
            (root / "test").mkdir()
            (root / "test" / "copy.png").write_bytes(
                (root / "train" / "a.png").read_bytes()
            )
            with self.assertRaisesRegex(ValueError, "leakage"):
                build_manifest(
                    {"train": root / "train", "test": root / "test"},
                    root / "manifest.jsonl",
                    dataset_root=root,
                )

    def test_manifest_admits_only_explicit_lossless_color_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_pattern(root / "train" / "rgb.png", 1)
            Image.new("L", (48, 48), 127).save(root / "train" / "gray.png")
            Image.new("RGB", (48, 48), (10, 20, 30)).save(
                root / "train" / "renamed.png", format="JPEG"
            )
            records = build_manifest(
                {"train": root / "train"},
                root / "manifest.jsonl",
                dataset_root=root,
            )
            self.assertEqual(
                [record.relative_path for record in records], ["train/rgb.png"]
            )
            metadata = json.loads(
                (root / "manifest.jsonl.meta.json").read_text(encoding="utf-8")
            )
            self.assertFalse(Path(metadata["dataset_root"]).is_absolute())
            reasons = {entry["reason"] for entry in metadata["skipped"]}
            self.assertTrue(any("not lossless" in reason for reason in reasons))
            self.assertTrue(any("4:4:4" in reason for reason in reasons))


class DegradationAndBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(4)
        self.rgb = generator.random((32, 40, 3), dtype=np.float32)
        self.target = rgb_to_ycrcb(self.rgb)

    def test_color_round_trip(self) -> None:
        reconstructed = ycrcb_to_rgb(self.target)
        self.assertLess(float(np.max(np.abs(reconstructed - self.rgb))), 0.002)

    def test_siting_filter_cross_product_and_baselines(self) -> None:
        for siting in ("center", "left", "cosited"):
            for down_filter in ("box", "gaussian", "lanczos3"):
                spec = DegradationSpec(
                    name=f"{siting}_{down_filter}",
                    siting=siting,
                    downsample_filter=down_filter,
                )
                model_input, low = simulate_420(self.target, spec)
                self.assertEqual(model_input.shape, self.target.shape)
                self.assertEqual(low.shape, (16, 20, 2))
                self.assertTrue(np.isfinite(model_input).all())
            _, low = simulate_420(
                self.target, DegradationSpec("baseline", siting=siting)
            )
            for method in CLASSICAL_METHODS:
                candidate = classical_reconstruction(
                    method, self.target[..., 0], low, siting
                )
                self.assertEqual(candidate.shape, self.target[..., 1:3].shape)
                self.assertTrue(np.isfinite(candidate).all())


class MetricTests(unittest.TestCase):
    def test_identity_metrics(self) -> None:
        image = np.random.default_rng(8).random((24, 24, 3), dtype=np.float32)
        metrics = reconstruction_metrics(image, image)
        self.assertEqual(metrics["rgb_psnr"], 80.0)
        self.assertEqual(metrics["chroma_psnr"], 80.0)
        self.assertAlmostEqual(metrics["rgb_ssim"], 1.0, places=10)
        self.assertAlmostEqual(metrics["delta_e2000_mean"], 0.0, places=10)

    def test_ciede2000_published_reference_pair(self) -> None:
        first = np.asarray([[50.0, 2.6772, -79.7751]])
        second = np.asarray([[50.0, 0.0, -82.7485]])
        value = float(delta_e_ciede2000_lab(first, second)[0])
        self.assertAlmostEqual(value, 2.0425, places=4)


class ModelAndCodecTests(unittest.TestCase):
    def test_ablation_contracts_and_losses(self) -> None:
        inputs = torch.rand(2, 3, 16, 20)
        target = torch.rand(2, 3, 16, 20)
        for config in (
            AblationConfig("base", depth=2, width=8),
            AblationConfig(
                "srcnn", architecture="srcnn", depth=3, width=8
            ),
            AblationConfig(
                "naf", architecture="naf_style", depth=2, width=8
            ),
            AblationConfig(
                "no_luma", luma_guidance=False, depth=1, width=8, loss="mse"
            ),
            AblationConfig(
                "direct",
                global_residual=False,
                depth=1,
                width=8,
                loss="charbonnier",
            ),
            AblationConfig(
                "edge", depth=1, width=8, loss="l1_edge", edge_weight=0.1
            ),
        ):
            model = build_ablation_model(config)
            output = model(inputs)
            self.assertEqual(output.shape, inputs.shape)
            self.assertTrue(torch.isfinite(chroma_loss(output, target, config)))
            self.assertTrue(torch.equal(output[:, 0], inputs[:, 0]))

    def test_all_learned_families_round_trip_checkpoints(self) -> None:
        inputs = torch.rand(1, 3, 12, 14)
        with tempfile.TemporaryDirectory() as temporary:
            for architecture in ("residual", "srcnn", "naf_style"):
                config = AblationConfig(
                    architecture, architecture=architecture, depth=1, width=8
                )
                model = build_ablation_model(config).eval()
                expected = model(inputs)
                checkpoint = Path(temporary) / f"{architecture}.pth"
                save_ablation_checkpoint(checkpoint, model, epoch=3)
                loaded, metadata = load_ablation_checkpoint(checkpoint)
                loaded.eval()
                self.assertEqual(metadata["format_version"], 2)
                self.assertEqual(metadata["config"]["architecture"], architecture)
                self.assertTrue(torch.equal(expected, loaded(inputs)))

    def test_actual_jpeg_roundtrip(self) -> None:
        rgb = np.random.default_rng(12).random((32, 32, 3), dtype=np.float32)
        result = jpeg_roundtrip(rgb, quality=60)
        self.assertEqual(result.rgb.shape, rgb.shape)
        self.assertGreater(result.encoded_bytes, 0)
        self.assertEqual(result.codec, "jpeg_420")

    def test_configured_video_codec_roundtrips(self) -> None:
        executable = os.environ.get("CHROMA_FFMPEG")
        if not executable:
            self.skipTest("CHROMA_FFMPEG is not configured")
        capabilities = ffmpeg_capabilities(executable)
        self.assertTrue(capabilities["available"])
        rgb = np.random.default_rng(21).random((32, 32, 3), dtype=np.float32)
        for codec, quality in (("h264", 23), ("hevc", 28), ("av1", 30)):
            result = video_roundtrip(rgb, codec, quality, executable)
            self.assertEqual(result.rgb.shape, rgb.shape)
            self.assertGreater(result.encoded_bytes, 0)
            self.assertEqual(result.codec, codec)


class EndToEndSmokeTest(unittest.TestCase):
    def test_benchmark_writes_per_image_and_qualitative_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split, offsets in (("train", (1, 2)), ("test", (3, 4))):
                for index, offset in enumerate(offsets):
                    write_pattern(root / "dataset" / split / f"{index}.png", offset)
            manifest = root / "manifest.jsonl"
            build_manifest(
                {
                    "train": root / "dataset" / "train",
                    "test": root / "dataset" / "test",
                },
                manifest,
                dataset_root=root / "dataset",
            )
            output = root / "report"
            configuration = {
                "manifest": str(manifest),
                "dataset_root": str(root / "dataset"),
                "output_dir": str(output),
                "split": "test",
                "crop_size": 32,
                "seed": 5,
                "device": "cpu",
                "profile_warmup": 0,
                "profile_repeats": 1,
                "bootstrap_samples": 100,
                "synthetic_degradations": [
                    {
                        "name": "center_box_bilinear",
                        "siting": "center",
                        "downsample_filter": "box",
                        "upsample_filter": "bilinear",
                    }
                ],
                "classical_methods": ["bilinear", "guided"],
                "learned_methods": [],
                "jpeg_qualities": [50],
                "video_codecs": [],
                "qualitative_count": 1,
                "qualitative_crop_size": 16,
            }
            report = run_benchmark(configuration, PROJECT_ROOT)
            self.assertEqual(report["evaluated_source_images"], 2)
            self.assertEqual(report["per_image_records"], 6)
            self.assertFalse(Path(report["provenance"]["manifest"]).is_absolute())
            self.assertTrue((output / "per_image.jsonl").is_file())
            self.assertTrue((output / "report.json").is_file())
            self.assertTrue((output / "qualitative" / "index.json").is_file())
            lines = (output / "per_image.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 6)
            self.assertIn("delta_e2000_mean", json.loads(lines[0])["metrics"])


class StaticProtocolConfigTests(unittest.TestCase):
    def test_full_configuration_covers_matrix_and_all_ablation_checkpoints(self) -> None:
        benchmark = json.loads(
            (PROJECT_ROOT / "research/configs/benchmark_full.json").read_text()
        )
        matrix = json.loads(
            (PROJECT_ROOT / "research/configs/ablations.json").read_text()
        )
        expected_degradations = {
            (siting, filter_name)
            for siting in ("center", "left", "cosited")
            for filter_name in ("box", "triangle", "gaussian", "lanczos3")
        }
        actual_degradations = {
            (entry["siting"], entry["downsample_filter"])
            for entry in benchmark["synthetic_degradations"]
        }
        self.assertEqual(actual_degradations, expected_degradations)
        self.assertEqual(
            {
                (entry["siting"], entry["downsample_filter"])
                for entry in matrix["degradations"]
            },
            expected_degradations,
        )
        experiment_names = {entry["name"] for entry in matrix["experiments"]}
        benchmark_checkpoints = {
            Path(entry["weights"]).parent.name
            for entry in benchmark["learned_methods"]
            if entry["type"] == "ablation"
        }
        self.assertEqual(benchmark_checkpoints, experiment_names)

        smoke = json.loads(
            (PROJECT_ROOT / "research/configs/benchmark_smoke.json").read_text()
        )
        self.assertEqual(
            {
                (entry["siting"], entry["downsample_filter"])
                for entry in smoke["synthetic_degradations"]
            },
            expected_degradations,
        )
        self.assertEqual(smoke["jpeg_qualities"], benchmark["jpeg_qualities"])
        self.assertEqual(smoke["video_codecs"], benchmark["video_codecs"])


if __name__ == "__main__":
    unittest.main()
