from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from chroma.checkpoints import load_model
from chroma.data import YUVChromaDataset, simulate_420, split_files
from chroma.inference import predict
from chroma.metrics import image_metrics
from chroma.models import ChromaRefiner, Discriminator, UNetGenerator, parameter_count
from chroma.training import TrainingConfig, _restore_training_state, run_training

ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def test_parameter_counts_are_stable(self) -> None:
        self.assertEqual(parameter_count(UNetGenerator()), 1_925_667)
        self.assertEqual(parameter_count(Discriminator()), 694_241)
        self.assertEqual(parameter_count(ChromaRefiner()), 593_794)

    def test_v5_shape_and_v6_luma_passthrough(self) -> None:
        inputs = torch.rand(1, 3, 70, 71)
        with torch.inference_mode():
            v5_output = predict(UNetGenerator(), inputs, "v5")
            v6_output = predict(ChromaRefiner(features=8, num_blocks=1), inputs, "v6")
        self.assertEqual(v5_output.shape, inputs.shape)
        self.assertEqual(v6_output.shape, inputs.shape)
        torch.testing.assert_close(v6_output[:, 0], inputs[:, 0])

    def test_committed_checkpoints_load(self) -> None:
        cases = (
            ("v5", UNetGenerator(), ROOT / "models/version5/epoch_010.pth"),
            ("v6", ChromaRefiner(), ROOT / "models/version6/res_epoch_030.pth"),
        )
        for version, model, path in cases:
            with self.subTest(version=version):
                checkpoint = load_model(model, path, version, "cpu")
                self.assertGreaterEqual(int(checkpoint["epoch"]), 1)

    def test_legacy_v5_checkpoint_resumes_at_epoch_11(self) -> None:
        generator = UNetGenerator()
        discriminator = Discriminator()
        checkpoint = load_model(
            generator, ROOT / "models/version5/epoch_010.pth", "v5", "cpu"
        )
        optimizers = {
            "generator": torch.optim.Adam(generator.parameters(), lr=2e-4),
            "discriminator": torch.optim.Adam(discriminator.parameters(), lr=2e-4),
        }
        start_epoch, best = _restore_training_state(
            checkpoint, "v5", discriminator, optimizers
        )
        self.assertEqual(start_epoch, 11)
        self.assertEqual(best, float("inf"))

    def test_new_checkpoint_round_trip(self) -> None:
        original = ChromaRefiner(features=8, num_blocks=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save(
                {"model_version": "v6", "epoch": 3, "model": original.state_dict()},
                path,
            )
            restored = ChromaRefiner(features=8, num_blocks=1)
            checkpoint = load_model(restored, path, "v6", "cpu")
        self.assertEqual(checkpoint["epoch"], 3)
        for left, right in zip(original.parameters(), restored.parameters()):
            torch.testing.assert_close(left, right)


class DataAndMetricTests(unittest.TestCase):
    def test_simulated_420_preserves_luma(self) -> None:
        image = np.random.default_rng(4).random((33, 41, 3), dtype=np.float32)
        degraded = simulate_420(image)
        self.assertEqual(degraded.shape, image.shape)
        np.testing.assert_array_equal(degraded[:, :, 0], image[:, :, 0])
        self.assertFalse(np.array_equal(degraded[:, :, 1:], image[:, :, 1:]))

    def test_split_is_deterministic_and_disjoint(self) -> None:
        files = [Path(f"image_{index}.png") for index in range(20)]
        first = split_files(files, 0.2, seed=7)
        second = split_files(files, 0.2, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 16)
        self.assertEqual(len(first[1]), 4)
        self.assertTrue(set(first[0]).isdisjoint(first[1]))

    def test_crop_validation(self) -> None:
        with self.assertRaises(ValueError):
            YUVChromaDataset([], crop_size=1, random_crop=True)

    def test_perfect_metrics(self) -> None:
        image = np.random.default_rng(8).random((32, 32, 3), dtype=np.float32)
        result = image_metrics(image, image)
        self.assertEqual(result["rgb_psnr"], 80.0)
        self.assertEqual(result["chroma_psnr"], 80.0)
        self.assertAlmostEqual(result["rgb_ssim"], 1.0, places=7)
        self.assertAlmostEqual(result["chroma_ssim"], 1.0, places=7)


class TrainingIntegrationTests(unittest.TestCase):
    def test_v5_and_v6_one_epoch_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            y, x = np.mgrid[:64, :64]
            for index in range(4):
                image = np.stack(
                    (
                        (x * 4 + index * 17) % 256,
                        (y * 4 + index * 31) % 256,
                        ((x + y) * 2 + index * 47) % 256,
                    ),
                    axis=2,
                ).astype(np.uint8)
                self.assertTrue(
                    cv2.imwrite(str(data_dir / f"image_{index}.png"), image)
                )

            for version in ("v5", "v6"):
                with self.subTest(version=version):
                    output_dir = root / version / "checkpoints"
                    samples_dir = root / version / "samples"
                    run_training(
                        TrainingConfig(
                            model_version=version,
                            source=data_dir,
                            output_dir=output_dir,
                            samples_dir=samples_dir,
                            epochs=1,
                            batch_size=1,
                            crop_size=64,
                            workers=0,
                            val_fraction=0.25,
                            seed=5,
                            device="cpu",
                            amp=False,
                            max_images=3,
                            run_name=f"{version}.test",
                        )
                    )
                    self.assertTrue((output_dir / "last.pth").is_file())
                    self.assertTrue((output_dir / "best.pth").is_file())
                    self.assertTrue((output_dir / "history.jsonl").is_file())
                    self.assertTrue((samples_dir / "epoch_001.png").is_file())
                    payload = torch.load(
                        output_dir / "last.pth", map_location="cpu", weights_only=True
                    )
                    self.assertEqual(payload["run_name"], f"{version}.test")
                    self.assertEqual(payload["config"]["max_images"], 3)


if __name__ == "__main__":
    unittest.main()
