import unittest

import torch

from chroma.research_data import DegradationSpec
from chroma.v7 import V7Config, V7PolarChromaRefiner
from chroma.v7_losses import degrade_chroma_torch
from chroma.v7_self_training import SelfTrainingConfig, make_pseudo_labels


class SelfTrainingTests(unittest.TestCase):
    def test_acceptance_requires_all_configured_conditions(self) -> None:
        teacher = V7PolarChromaRefiner(V7Config(width=8, depth=1))
        inputs = torch.rand(1, 3, 8, 8)
        spec = DegradationSpec("center_box", "center", "box", "bilinear")
        observed = degrade_chroma_torch(inputs[:, 1:3], spec)
        permissive = SelfTrainingConfig(
            enabled=True,
            amplitude_scale_max=1.0,
            phase_kappa_min=0.1,
            forward_error_max=1.0,
            augmentation_error_max=1.0,
            minimum_acceptance_confidence=1e-4,
        )
        pseudo, weights, stats = make_pseudo_labels(
            teacher, inputs, observed, [spec], permissive
        )
        self.assertEqual(pseudo.shape, inputs[:, 1:3].shape)
        self.assertGreater(stats["accepted_pixels"], 0.0)
        self.assertGreater(float(weights.sum()), 0.0)
        restrictive = SelfTrainingConfig(
            enabled=True,
            amplitude_scale_max=1e-6,
            phase_kappa_min=99.0,
            forward_error_max=1e-6,
            augmentation_error_max=1e-6,
            minimum_acceptance_confidence=0.99,
        )
        _, weights, stats = make_pseudo_labels(
            teacher, inputs, observed, [spec], restrictive
        )
        self.assertEqual(stats["accepted_pixels"], 0.0)
        self.assertEqual(float(weights.sum()), 0.0)

    def test_flip_safeguard_rejects_non_center_siting(self) -> None:
        teacher = V7PolarChromaRefiner(V7Config(width=8, depth=1))
        inputs = torch.rand(1, 3, 8, 8)
        spec = DegradationSpec("left_box", "left", "box", "bilinear")
        observed = degrade_chroma_torch(inputs[:, 1:3], spec)
        with self.assertRaises(ValueError):
            make_pseudo_labels(
                teacher,
                inputs,
                observed,
                [spec],
                SelfTrainingConfig(enabled=True),
            )


if __name__ == "__main__":
    unittest.main()
