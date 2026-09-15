import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chroma.model_comparison import compare, METRICS, summarize
from chroma.prism_config import load_suite
from chroma.research_data import sha256_file

ROOT = Path(__file__).resolve().parents[1]


class ModelComparisonTests(unittest.TestCase):
    def test_bilinear_identity_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "test.jsonl"
            rows = [json.loads(line) for line in (ROOT / "research/manifests/fixture.jsonl").read_text().splitlines()]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows if row["split"] == "test"))
            weights = root / "dummy.pth"
            weights.write_bytes(b"fixture")
            suite = load_suite(ROOT / "research/configs/prism_smoke.json")
            suite["degradations"] = suite["degradations"][:2]
            campaign = root / "campaign.json"
            campaign.write_text(json.dumps({"suite": suite, "manifest": str(manifest), "models": [
                {"name": "identity", "kind": "legacy", "group": 0, "weights": str(weights),
                 "sha256": sha256_file(weights)}]}))
            with patch("chroma.model_comparison.load_candidate", return_value=(torch.nn.Identity(), {})):
                first = compare(campaign, ROOT / "research/fixtures/lossless", root / "output",
                                device="cpu", batch_size=1, workers=0, limit=1)
                # Emulate interruption after one committed sample; later writes may be incomplete.
                progress_path = root / "output/progress.json"
                progress = json.loads(progress_path.read_text())
                progress["completed_samples"] = 1
                progress_path.write_text(json.dumps(progress))
                partial = np.load(root / "output/metrics.npy", mmap_mode="r+")
                partial[1:] = np.nan
                partial.flush()
                second = compare(campaign, ROOT / "research/fixtures/lossless", root / "output",
                                 device="cpu", batch_size=1, workers=0, limit=1)
            self.assertEqual(first["results"], second["results"])
            values = np.load(root / "output/metrics.npy")
            np.testing.assert_array_equal(values[:, 0], values[:, 1])
            self.assertEqual(first["samples"], len(suite["degradations"]))
            for value in first["results"]["identity"]["metrics"].values():
                self.assertEqual(value["mean_gain_vs_bilinear"], 0)

    def test_collector_requires_complete_matching_campaign(self):
        from scripts.collect_model_comparison import collect
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign_path = root / "campaign.json"
            campaign_path.write_text(json.dumps({"models": [{"name": "model", "group": 0}], "images": 1}))
            self.assertFalse(collect(root))
            path = root / "groups/0/report.json"
            path.parent.mkdir(parents=True)
            values = np.zeros((1, 2, len(METRICS)), dtype=np.float32)
            report = {"status": "complete", "images": 1, "campaign_sha256": sha256_file(campaign_path),
                      "models": [{"name": "model"}],
                      "results": summarize(values, ["bilinear", "model"], ["test"])}
            path.write_text(json.dumps(report))
            self.assertTrue(collect(root))
            self.assertIn("model", (root / "comparison.csv").read_text())
            report["campaign_sha256"] = "wrong"
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                collect(root)

    def test_paired_gain_direction_and_condition_indexing(self):
        values = np.zeros((4, 2, len(METRICS)), dtype=np.float32)
        values[:, 0, METRICS.index("chroma_l1")] = 2
        values[::2, 1, METRICS.index("chroma_psnr")] = 3
        result = summarize(values, ["bilinear", "model"], ["first", "second"])["model"]
        self.assertEqual(result["metrics"]["chroma_l1"]["mean_gain_vs_bilinear"], 2)
        self.assertEqual(result["by_degradation"]["first"]["chroma_psnr"]["mean_gain_vs_bilinear"], 3)
        self.assertEqual(result["by_degradation"]["second"]["chroma_psnr"]["mean_gain_vs_bilinear"], 0)
