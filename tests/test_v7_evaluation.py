import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from chroma.research_baselines import load_v7_predictor
from chroma.v7 import V7Config, V7PolarChromaRefiner, save_v7_checkpoint
from chroma.v7_evaluation import run_v7_evaluation
from chroma.v7_losses import V7LossConfig


class V7EvaluationSmokeTests(unittest.TestCase):
    def test_fixture_writes_mean_safe_maps_and_diagnostics(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "v7.pth"
            save_v7_checkpoint(
                weights,
                V7PolarChromaRefiner(V7Config(width=8, depth=1)),
                epoch=0,
                seed=1,
                loss_config=asdict(V7LossConfig()),
            )
            report = run_v7_evaluation(
                {
                    "manifest": "research/manifests/fixture.jsonl",
                    "dataset_root": "research/fixtures/lossless",
                    "weights": str(weights),
                    "output_dir": str(root / "report"),
                    "split": "test",
                    "crop_size": 64,
                    "limit": 1,
                    "device": "cpu",
                    "save_pixel_arrays": True,
                    "qualitative_count": 1,
                    "qualitative_crop_size": 32,
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
            self.assertEqual(report["records"], 2)
            lines = (root / "report" / "per_image.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(any((root / "report" / "pixel_maps").glob("*.npz")))
            self.assertTrue(any((root / "report" / "qualitative").glob("*.png")))
            loaded, predictor, metadata = load_v7_predictor(
                weights, torch.device("cpu"), "safe"
            )
            self.assertEqual(metadata["family"], "v7")
            self.assertEqual(metadata["mode"], "safe")
            self.assertEqual(loaded.config.width, 8)
            output = predictor(np.full((8, 8, 3), 0.5, dtype=np.float32))
            self.assertEqual(output.shape, (8, 8, 3))


if __name__ == "__main__":
    unittest.main()
