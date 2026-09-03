import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from chroma.models import ChromaRefiner
from chroma.models import parameter_count as legacy_parameter_count
from chroma.v7 import (
    V7Config,
    V7PolarChromaRefiner,
    load_v7_checkpoint,
    parameter_count,
    save_v7_checkpoint,
)
from chroma.v7_training import train_v7


class V7ModelTests(unittest.TestCase):
    def test_ablation_matrix_contains_matched_v6_and_all_v7_stages(self) -> None:
        project = Path(__file__).resolve().parents[1]
        v6 = json.loads((project / "research/configs/ablations.json").read_text())
        matrix = json.loads(
            (project / "research/configs/v7_ablations.json").read_text()
        )
        for name in (
            "epochs",
            "batch_size",
            "crop_size",
            "workers",
            "learning_rate",
            "weight_decay",
            "seed",
        ):
            self.assertEqual(matrix["training"][name], v6["training"][name])
        self.assertEqual(matrix["degradations"], v6["degradations"])
        self.assertEqual(
            {entry["name"] for entry in matrix["controls"]},
            {"v6_cartesian_deterministic"},
        )
        self.assertEqual(
            {entry["name"] for entry in matrix["experiments"]},
            {
                "v7_polar_deterministic",
                "v7_probabilistic",
                "v7_probabilistic_forward",
            },
        )
        self.assertEqual(
            matrix["inference_variants"][0]["name"], "v7_probabilistic_safe"
        )
        self.assertEqual(matrix["second_stage_variants"][0]["name"], "v7_self_train")

    def test_parameter_count_and_identity_initialization(self) -> None:
        model = V7PolarChromaRefiner()
        self.assertEqual(legacy_parameter_count(ChromaRefiner()), 593_794)
        self.assertEqual(parameter_count(model), 595_525)
        inputs = torch.rand(2, 3, 12, 14)
        prediction = model(inputs)
        torch.testing.assert_close(
            prediction.ycrcb("mean")[:, 1:3], inputs[:, 1:3], atol=3e-6, rtol=3e-6
        )
        torch.testing.assert_close(prediction.luma, inputs[:, 0:1])
        self.assertTrue(torch.isfinite(prediction.raw).all())

    def test_forward_backward_and_extreme_uncertainty(self) -> None:
        model = V7PolarChromaRefiner(V7Config(width=8, depth=1))
        inputs = torch.rand(2, 3, 8, 8)
        with torch.no_grad():
            model.tail.bias[1] = 0.0
            model.tail.bias[2] = 0.0
            model.tail.bias[3] = -100.0
            model.tail.bias[4] = 1000.0
        prediction = model(inputs)
        self.assertTrue(torch.isfinite(prediction.amplitude_scale).all())
        self.assertTrue(torch.isfinite(prediction.phase_kappa).all())
        self.assertLessEqual(
            float(prediction.phase_kappa.max().detach()), model.config.kappa_max
        )
        prediction.ycrcb().mean().backward()
        self.assertTrue(
            all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )

    def test_checkpoint_round_trip_and_strict_metadata(self) -> None:
        config = V7Config(width=8, depth=2, neutral_chroma=0.5)
        model = V7PolarChromaRefiner(config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "v7.pth"
            save_v7_checkpoint(
                path,
                model,
                epoch=3,
                seed=17,
                loss_config={"lambda_cart": 1.0},
                git_commit="test",
            )
            restored, metadata = load_v7_checkpoint(path, expected_config=config)
            self.assertEqual(metadata["training_seed"], 17)
            self.assertEqual(metadata["architecture"], asdict(config))
            for first, second in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(first, second)
            with self.assertRaises(ValueError):
                load_v7_checkpoint(path, expected_config=V7Config(width=16, depth=2))

    def test_safe_mode_backs_off_under_high_uncertainty(self) -> None:
        model = V7PolarChromaRefiner(V7Config(width=8, depth=1))
        with torch.no_grad():
            model.tail.weight.zero_()
            model.tail.bias[0] = 0.1
            model.tail.bias[1] = 0.0
            model.tail.bias[2] = 1.0
            model.tail.bias[3] = 20.0
            model.tail.bias[4] = -20.0
        inputs = torch.full((1, 3, 8, 8), 0.6)
        prediction = model(inputs)
        baseline = inputs[:, 1:3]
        mean_change = torch.mean(torch.abs(prediction.chroma_mean - baseline))
        safe_change = torch.mean(torch.abs(prediction.chroma_safe - baseline))
        self.assertLess(float(safe_change.detach()), float(mean_change.detach()) * 0.01)

    def test_manifest_training_pipeline_one_epoch_smoke(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "v7_training"
            result = train_v7(
                {
                    "manifest": "research/manifests/fixture.jsonl",
                    "dataset_root": "research/fixtures/lossless",
                    "output_dir": str(output),
                    "model": {"width": 8, "depth": 1, "neutral_chroma": 0.5},
                    "loss": {
                        "lambda_cart": 1.0,
                        "lambda_amplitude": 0.1,
                        "lambda_phase": 0.05,
                        "lambda_forward": 0.1,
                        "phase_reference_amplitude": 0.05,
                        "probabilistic": True,
                        "debug_finite": True,
                    },
                    "training": {
                        "crop_size": 64,
                        "batch_size": 2,
                        "epochs": 1,
                        "workers": 0,
                        "learning_rate": 1e-3,
                        "seed": 9,
                        "device": "cpu",
                        "amp": False,
                    },
                    "degradations": [
                        {
                            "name": "center_box",
                            "siting": "center",
                            "downsample_filter": "box",
                            "upsample_filter": "bilinear",
                        }
                    ],
                },
                project,
            )
            self.assertEqual(len(result["history"]), 1)
            self.assertTrue((output / "last.pth").is_file())
            self.assertTrue((output / "best.pth").is_file())
            _, checkpoint = load_v7_checkpoint(output / "last.pth")
            self.assertEqual(checkpoint["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
