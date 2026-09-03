import math
import unittest

import numpy as np
import torch

from chroma.chroma_polar import (
    OPENCV_UINT8_NEUTRAL_CHROMA,
    PUBLICATION_NEUTRAL_CHROMA,
    cartesian_chroma_to_polar,
    circular_difference,
    polar_chroma_to_cartesian,
    wrap_angle,
)
from chroma.data import rgb_to_ycrcb as legacy_rgb_to_ycrcb
from chroma.research_data import rgb_to_ycrcb as publication_rgb_to_ycrcb


class PolarConversionTests(unittest.TestCase):
    neutral = 0.5

    def assert_round_trip(self, chroma: torch.Tensor) -> None:
        amplitude, phase = cartesian_chroma_to_polar(chroma, self.neutral)
        restored = polar_chroma_to_cartesian(amplitude, phase, self.neutral)
        torch.testing.assert_close(restored, chroma, atol=2e-7, rtol=2e-7)

    def test_random_and_boundary_round_trip(self) -> None:
        generator = torch.Generator().manual_seed(2026)
        self.assert_round_trip(torch.rand((4, 2, 17, 19), generator=generator))
        boundary = torch.tensor([0.0, 0.5, 1.0])
        cr, cb = torch.meshgrid(boundary, boundary, indexing="ij")
        self.assert_round_trip(torch.stack((cr, cb)).unsqueeze(0))

    def test_repository_neutral_conventions_are_explicit(self) -> None:
        legacy_gray = legacy_rgb_to_ycrcb(np.full((1, 1, 3), 128, dtype=np.uint8))
        publication_gray = publication_rgb_to_ycrcb(
            np.full((1, 1, 3), 0.5, dtype=np.float32)
        )
        self.assertAlmostEqual(float(legacy_gray[0, 0, 1]), OPENCV_UINT8_NEUTRAL_CHROMA)
        self.assertAlmostEqual(float(legacy_gray[0, 0, 2]), OPENCV_UINT8_NEUTRAL_CHROMA)
        self.assertEqual(float(publication_gray[0, 0, 1]), PUBLICATION_NEUTRAL_CHROMA)
        self.assertEqual(float(publication_gray[0, 0, 2]), PUBLICATION_NEUTRAL_CHROMA)

    def test_neutral_and_near_zero(self) -> None:
        neutral = torch.full((2, 2, 3, 3), self.neutral)
        amplitude, phase = cartesian_chroma_to_polar(neutral, self.neutral)
        self.assertTrue(torch.equal(amplitude, torch.zeros_like(amplitude)))
        self.assertTrue(torch.equal(phase, torch.zeros_like(phase)))
        self.assert_round_trip(neutral)
        near = neutral.clone()
        near[:, 0] += 1e-7
        near[:, 1] -= 1e-7
        self.assert_round_trip(near)

    def test_all_quadrants(self) -> None:
        chroma = torch.tensor(
            [[[[0.75, 0.25, 0.25, 0.75]], [[0.75, 0.75, 0.25, 0.25]]]]
        )
        _, phase = cartesian_chroma_to_polar(chroma, self.neutral)
        expected = torch.tensor(
            [math.pi / 4, 3 * math.pi / 4, -3 * math.pi / 4, -math.pi / 4]
        )
        torch.testing.assert_close(phase.flatten(), expected)
        self.assert_round_trip(chroma)

    def test_wrap_and_circular_difference_at_branch_cut(self) -> None:
        values = torch.tensor([-3 * math.pi, -math.pi, math.pi, 3 * math.pi])
        wrapped = wrap_angle(values)
        self.assertTrue(torch.all(wrapped >= -math.pi))
        self.assertTrue(torch.all(wrapped < math.pi))
        difference = circular_difference(
            torch.tensor([-math.pi + 1e-4]), torch.tensor([math.pi - 1e-4])
        )
        self.assertAlmostEqual(float(difference), 2e-4, places=5)


if __name__ == "__main__":
    unittest.main()
