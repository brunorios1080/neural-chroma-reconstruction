import unittest

import numpy as np
import torch

from chroma.chroma_polar import circular_difference
from chroma.research_data import DegradationSpec, simulate_420
from chroma.v7_fixtures import (
    ambiguous_same_luma_pair,
    hue_wrap_pair,
    neutral_fixture,
    sharp_chroma_edge,
)


class HypothesisFixtureTests(unittest.TestCase):
    def test_different_chroma_can_have_identical_observation(self) -> None:
        first, second = ambiguous_same_luma_pair(16)
        self.assertTrue(np.array_equal(first[..., 0], second[..., 0]))
        self.assertGreater(float(np.max(np.abs(first[..., 1:] - second[..., 1:]))), 0.1)
        spec = DegradationSpec("center_box", "center", "box", "bilinear")
        first_input, first_low = simulate_420(first, spec)
        second_input, second_low = simulate_420(second, spec)
        np.testing.assert_allclose(first_low, second_low, atol=1e-7)
        np.testing.assert_allclose(first_input, second_input, atol=1e-7)

    def test_neutral_wrap_and_edge_fixtures(self) -> None:
        neutral = neutral_fixture(16)
        np.testing.assert_array_equal(neutral[..., 1:], np.full((16, 16, 2), 0.5))
        first, second = hue_wrap_pair()
        first_phase = torch.atan2(
            torch.tensor(first[0, 0, 2] - 0.5), torch.tensor(first[0, 0, 1] - 0.5)
        )
        second_phase = torch.atan2(
            torch.tensor(second[0, 0, 2] - 0.5), torch.tensor(second[0, 0, 1] - 0.5)
        )
        self.assertLess(
            float(circular_difference(first_phase, second_phase).abs()), 3e-4
        )
        edge = sharp_chroma_edge(16)
        self.assertGreater(float(np.max(np.abs(np.diff(edge[..., 1], axis=1)))), 0.4)


if __name__ == "__main__":
    unittest.main()
