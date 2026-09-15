"""Behavioral checks for the new family, data sampling, and resumable training."""

import copy
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from chroma.prism_config import load_suite, experiment_config
from chroma.prism_data import PrismDataset, EpochSampler, audit_manifest
from chroma.prism_evaluation import evaluate_prism
from chroma.prism_metrics import psnr_ssim
from chroma.prism_models import PrismArchitecture, PrismRefiner
from chroma.prism_training import (
    train_prism,
    load_checkpoint,
    supervised_loss,
    measurement_loss,
)
from chroma.research_data import load_manifest, DegradationSpec
from chroma.research_metrics import ssim

ROOT = Path(__file__).resolve().parents[1]


class PrismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)

    def suite(self, destination):
        suite = load_suite(ROOT / "research/configs/prism_smoke.json")
        suite["output_dir"] = str(destination)
        suite["model"].update(width=4, depth=1)
        for experiment in suite["experiments"]:
            experiment.setdefault("model", {}).update(width=4, depth=1)
        suite["degradations"] = [suite["degradations"][0], suite["degradations"][5]]
        return suite

    def test_bilinear_only_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = self.suite(Path(temporary) / "runs")
            suite["evaluation"] = {"baselines": ["bilinear"]}
            output = Path(temporary) / "evaluation"
            report = evaluate_prism(
                suite, [], ROOT / "research/manifests/fixture.jsonl",
                ROOT / "research/fixtures/lossless", output,
                limit=1, bootstrap_samples=10,
            )
            self.assertEqual(report["baseline_methods"], ["bilinear"])
            rows = [json.loads(line) for line in (output / "per_image.jsonl").read_text().splitlines()]
            self.assertEqual({row["method"] for row in rows}, {"bilinear"})
            self.assertEqual(len(rows), len(suite["degradations"]))

    def test_ssim_black_gray_and_random_match_reference(self):
        for level in (0.0, 0.001, 0.5):
            image = torch.full((1, 3, 16, 16), level)
            psnr, score = psnr_ssim(image, image)
            self.assertAlmostEqual(score.item(), 1.0, places=5)
            self.assertEqual(psnr.item(), 80.0)
        torch.manual_seed(1)
        first = torch.rand(1, 3, 16, 16)
        second = (first + 0.01).clamp(0, 1)
        expected = ssim(
            first[0].permute(1, 2, 0).numpy(), second[0].permute(1, 2, 0).numpy()
        )
        self.assertAlmostEqual(psnr_ssim(first, second)[1].item(), expected, places=5)

    def test_persistent_workers_refresh_crops_and_match_single_process(self):
        records = load_manifest(ROOT / "research/manifests/fixture.jsonl")[:3]
        specs = [
            DegradationSpec("box"),
            DegradationSpec("triangle", downsample_filter="triangle"),
        ]
        dataset = PrismDataset(
            records, ROOT / "research/fixtures/lossless", 32, specs, 2026, training=True
        )
        sampler = EpochSampler(dataset, 2026)
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=1,
            persistent_workers=True,
            multiprocessing_context="fork",
        )
        snapshots = []
        for epoch in (1, 2):
            sampler.set_epoch(epoch)
            expected = {dataset[key][-1]: dataset[key] for key in sampler}
            actual = {}
            for inputs, targets, low, indices, ids in loader:
                torch.testing.assert_close(targets[0], expected[ids[0]][1])
                self.assertEqual(indices.item(), expected[ids[0]][3])
                actual[ids[0]] = targets[0]
            snapshots.append(actual)
        self.assertTrue(
            any(
                not torch.equal(snapshots[0][key], snapshots[1][key])
                for key in snapshots[0]
            )
        )
        del loader

    def test_polar_radius_has_gradient_after_negative_overshoot(self):
        model = PrismRefiner(
            PrismArchitecture(width=4, depth=1, representation="polar")
        )
        with torch.no_grad():
            model.tail.bias[0] = -0.2
        inputs = torch.full((1, 3, 8, 8), 0.5)
        inputs[:, 1] = 0.6
        prediction = model.predict(inputs)
        prediction.amplitude.mean().backward()
        self.assertGreater(model.tail.bias.grad[0].item(), 0)
        self.assertGreater(prediction.amplitude.min().item(), 0)
        self.assertTrue(
            all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )

    def test_detached_uncertainty_does_not_change_reconstruction_gradients(self):
        architecture = PrismArchitecture(
            width=4, depth=1, uncertainty=True, detached_uncertainty=True
        )
        model = PrismRefiner(architecture)
        inputs = torch.rand(1, 3, 16, 16) * 0.5 + 0.25
        target = inputs.clone()
        target[:, 1] += 0.02
        gradients = []
        for weight in (0.0, 0.01):
            model.zero_grad(set_to_none=True)
            config = {
                "amplitude_weight": weight,
                "phase_weight": weight,
                "uncertainty_warmup_epochs": 0,
                "uncertainty_ramp_epochs": 1,
            }
            prediction = model.predict(inputs)
            total, _ = supervised_loss(
                prediction, target, None, None, [], config, 1, architecture
            )
            total.backward()
            gradients.append(model.tail.bias.grad.clone())
        torch.testing.assert_close(*gradients, rtol=0, atol=0)
        self.assertGreater(model.uncertainty_head.bias.grad.abs().sum().item(), 0)
        confidence = model.predict(inputs).confidence()
        self.assertTrue(torch.isfinite(confidence).all())
        self.assertTrue(((confidence >= 0) & (confidence <= 1)).all())

    def test_forward_operator_matches_dataset_and_conditioning_is_required(self):
        records = load_manifest(ROOT / "research/manifests/fixture.jsonl")[:1]
        specs = [
            DegradationSpec(
                "left_triangle", siting="left", downsample_filter="triangle"
            )
        ]
        dataset = PrismDataset(
            records, ROOT / "research/fixtures/lossless", 32, specs, 1
        )
        inputs, targets, low, _, _ = dataset[0]
        loss = measurement_loss(targets[None, 1:3], low[None], torch.tensor([0]), specs)
        self.assertLess(loss.item(), 1e-6)
        model = PrismRefiner(PrismArchitecture(width=4, depth=1, conditioned=True))
        with self.assertRaisesRegex(ValueError, "known degradation"):
            model(inputs[None])
        prediction = model(inputs[None], specs)
        torch.testing.assert_close(prediction, inputs[None])

    def test_cross_split_hash_leakage_is_rejected(self):
        records = load_manifest(ROOT / "research/manifests/fixture.jsonl")[:2]
        records[1] = replace(records[1], sha256=records[0].sha256, split="train")
        with self.assertRaisesRegex(ValueError, "Cross-split"):
            audit_manifest(
                records, ROOT / "research/fixtures/lossless", "lossless", 32, False
            )

    def test_all_twelve_experiments_train_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite = self.suite(Path(temporary) / "runs")
            self.assertEqual(len(suite["experiments"]), 12)
            for entry in suite["experiments"]:
                with self.subTest(model=entry["name"]):
                    config = experiment_config(suite, entry["name"])
                    first = train_prism(config, ROOT, stop_after_epoch=1)
                    self.assertEqual(first["completed_epochs"], 1)
                    resumed = train_prism(config, ROOT, resume="auto")
                    model, checkpoint = load_checkpoint(resumed["last_checkpoint"])
                    self.assertEqual(checkpoint["epoch"], 2)
                    self.assertEqual(checkpoint["metrics"]["validation"]["luma_l1"], 0)
                    self.assertLess(
                        checkpoint["metrics"]["learning_rate"],
                        config["training"]["learning_rate"],
                    )
                    self.assertTrue(Path(resumed["best_checkpoint"]).is_file())
                    history = Path(config["output_dir"]) / "history.jsonl"
                    self.assertEqual(
                        [
                            json.loads(s)["epoch"]
                            for s in history.read_text().splitlines()
                        ],
                        [1, 2],
                    )
                    with self.assertRaises(FileExistsError):
                        train_prism(config, ROOT)
                    changed = copy.deepcopy(config)
                    changed["loss"]["edge_weight"] += 0.1
                    with self.assertRaisesRegex(ValueError, "differs"):
                        train_prism(changed, ROOT, resume="auto")
            # Compare all trained models, including the conditioned model, on identical test inputs.
            weights = [
                Path(suite["output_dir"]) / entry["name"] / "best.pth"
                for entry in suite["experiments"]
            ]
            report = evaluate_prism(
                suite,
                weights,
                ROOT / "research/manifests/fixture.jsonl",
                ROOT / "research/fixtures/lossless",
                Path(temporary) / "report",
                limit=1,
                bootstrap_samples=10,
            )
            self.assertEqual(len(report["models"]), 12)
            self.assertTrue(report["paired_vs_bilinear"])

    def test_resume_matches_uninterrupted_training(self):
        for name in ("prism_residual", "prism_gan", "prism_cartesian_prob"):
            with self.subTest(model=name), tempfile.TemporaryDirectory() as temporary:
                continuous = experiment_config(
                    self.suite(Path(temporary) / "continuous"), name
                )
                interrupted = experiment_config(
                    self.suite(Path(temporary) / "interrupted"), name
                )
                full = train_prism(continuous, ROOT)
                train_prism(interrupted, ROOT, stop_after_epoch=1)
                resumed = train_prism(interrupted, ROOT, resume="auto")
                _, a = load_checkpoint(full["last_checkpoint"])
                _, b = load_checkpoint(resumed["last_checkpoint"])
                for key in a["model"]:
                    torch.testing.assert_close(
                        a["model"][key], b["model"][key], rtol=0, atol=0
                    )
                self.assertEqual(a["scheduler"], b["scheduler"])
                if "discriminator" in a:
                    for key in a["discriminator"]:
                        torch.testing.assert_close(
                            a["discriminator"][key],
                            b["discriminator"][key],
                            rtol=0,
                            atol=0,
                        )

    def test_submit_dry_run_maps_selected_models_without_submission(self):
        process = subprocess.run(
            [
                "bash",
                "scripts/bridges2/submit_prism.sh",
                "--dry-run",
                "prism_residual",
                "prism_gan",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("Array task 0: prism_residual", process.stdout)
        self.assertIn("--array=0-1%2", process.stdout)
        self.assertIn("--gres=gpu:l40s-48:1", process.stdout)

    def test_coco_preparation_deduplicates_before_splitting(self):
        from scripts.prepare_prism_coco import prepare

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "unlabeled2017"
            images.mkdir()
            random = np.random.default_rng(7)
            for index in range(50):
                image = random.integers(0, 256, (16, 16, 3), dtype=np.uint8)
                Image.fromarray(image).save(images / f"{index:012d}.jpg")
            (images / "duplicate.jpg").write_bytes(
                (images / "000000000000.jpg").read_bytes()
            )
            records = prepare(images, root, root / "first.jsonl", min_size=8)
            repeated = prepare(images, root, root / "second.jsonl", min_size=8)
            self.assertEqual(records, repeated)
            self.assertEqual(len(records), 50)
            self.assertEqual(len({r.sha256 for r in records}), 50)
            self.assertEqual(
                {r.split for r in records}, {"train", "validation", "test"}
            )
            audit_manifest(records, root, "synthetic_coco", 8)

    def test_cli_training_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = [
                sys.executable,
                "-B",
                "scripts/train_prism.py",
                "--config",
                "research/configs/prism_smoke.json",
                "--model",
                "prism_polar_prob,prism_gan",
                "--output-root",
                str(root / "runs"),
                "--workers",
                "1",
                "--stop-after-epoch",
                "1",
            ]
            trained = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(trained.returncode, 0, trained.stderr)
            self.assertTrue((root / "runs/prism_gan/last.pth").is_file())
            evaluated = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "scripts/evaluate_prism.py",
                    "--config",
                    "research/configs/prism_smoke.json",
                    "--weights",
                    str(root / "runs/prism_polar_prob/best.pth"),
                    str(root / "runs/prism_gan/best.pth"),
                    "--manifest",
                    "research/manifests/fixture.jsonl",
                    "--dataset-root",
                    "research/fixtures/lossless",
                    "--output-dir",
                    str(root / "report"),
                    "--device",
                    "cpu",
                    "--limit",
                    "1",
                    "--bootstrap-samples",
                    "10",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            self.assertTrue((root / "report/report.json").is_file())

    def test_detached_uncertainty_training_preserves_mean_with_large_auxiliary_gradients(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            suite = self.suite(Path(temporary))
            baseline = experiment_config(suite, "prism_residual")
            auxiliary = experiment_config(suite, "prism_cartesian_prob")
            auxiliary["loss"].update(amplitude_weight=1000.0, phase_weight=1000.0)
            full = train_prism(baseline, ROOT)
            separate = train_prism(auxiliary, ROOT)
            _, first = load_checkpoint(full["last_checkpoint"])
            _, second = load_checkpoint(separate["last_checkpoint"])
            for key, value in first["model"].items():
                torch.testing.assert_close(value, second["model"][key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
