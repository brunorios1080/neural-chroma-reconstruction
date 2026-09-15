import json
import tempfile
import unittest
from pathlib import Path

from scripts.review_prism import CORE, review


class PrismReviewTests(unittest.TestCase):
    def test_missing_runs_are_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            result = review(directory)
            self.assertFalse(result["ready_for_comparison"])
            self.assertEqual(len(result["runs"]), 4)

    def test_completed_matching_runs_and_partial_append(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in CORE:
                run = Path(directory) / name
                run.mkdir()
                row = {
                    "epoch": 25,
                    "epochs": 100,
                    "validation": {
                        "chroma_l1": 0.004,
                        "rgb_psnr": 40.0,
                        "rgb_ssim": 0.97,
                        "chroma_psnr_gain_db": 2.0,
                    },
                }
                (run / "history.jsonl").write_text(json.dumps(row) + '\n{"epoch":')
                (run / "manifest.jsonl").write_text("same manifest\n")
            result = review(directory)
            self.assertTrue(result["ready_for_comparison"])
            self.assertEqual(result["runs"][0]["best_epoch"], 25)
            (Path(directory) / CORE[0] / "manifest.jsonl").write_text("different\n")
            self.assertFalse(review(directory)["ready_for_comparison"])


if __name__ == "__main__":
    unittest.main()
