from __future__ import annotations

import unittest

import numpy as np

from chroma.metrics import reconstruction_metrics
from chroma.tabpfn import area_downsample, make_features
from scripts.benchmark_coco_tabpfn import (
    METHODS,
    compare_methods,
    directional_delta,
    summarize,
)


class TabPFNFeatureTests(unittest.TestCase):
    def test_feature_grid_and_downsampling(self) -> None:
        luma = np.random.default_rng(12).random((16, 20), dtype=np.float32)
        features = make_features(luma)
        low_features = area_downsample(features)
        self.assertEqual(features.shape, (16, 20, 16))
        self.assertEqual(low_features.shape, (8, 10, 16))
        self.assertTrue(np.isfinite(features).all())

    def test_extended_metrics_are_perfect_for_identity(self) -> None:
        image = np.random.default_rng(13).random((16, 16, 3), dtype=np.float32)
        scores = reconstruction_metrics(image, image)
        self.assertEqual(scores["chroma_mae"], 0.0)
        self.assertEqual(scores["chroma_edge_mae"], 0.0)
        self.assertEqual(scores["chroma_gradient_mae"], 0.0)


class BenchmarkStatisticsTests(unittest.TestCase):
    def test_directional_deltas_favor_better_values(self) -> None:
        self.assertEqual(directional_delta("chroma_psnr", 32.0, 30.0), 2.0)
        self.assertEqual(directional_delta("chroma_mae", 0.1, 0.2), 0.1)

    def test_paired_summary(self) -> None:
        records = []
        for offset in (0.0, 1.0):
            metrics = {}
            for index, method in enumerate(METHODS):
                metrics[method] = {
                    "chroma_psnr": 30.0 + offset + index,
                    "chroma_mae": 0.20 + offset * 0.01 - index * 0.01,
                }
            records.append({"metrics": metrics})
        aggregate, paired = summarize(records, bootstrap_samples=100, seed=4)
        self.assertEqual(aggregate["bilinear"]["chroma_psnr"]["mean"], 30.5)
        self.assertEqual(
            paired["tabpfn_v3"]["chroma_psnr"]["mean_improvement"], 4.0
        )
        self.assertEqual(paired["tabpfn_v3"]["chroma_mae"]["win_rate"], 1.0)
        comparison = compare_methods(
            records, "tabpfn_v3", "v6", bootstrap_samples=100, seed=4
        )
        self.assertEqual(comparison["chroma_psnr"]["mean_improvement"], 1.0)


if __name__ == "__main__":
    unittest.main()
