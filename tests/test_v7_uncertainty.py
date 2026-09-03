import math
import unittest

import numpy as np
import torch

from chroma.v7_uncertainty import (
    VonMisesIntervalLookup,
    circular_variance,
    empirical_interval_coverage,
    laplace_interval,
    risk_coverage_curve,
    spearman_correlation,
)


class UncertaintyTests(unittest.TestCase):
    def test_laplace_interval_formula(self) -> None:
        mean, scale = torch.tensor([0.3]), torch.tensor([0.1])
        lower, upper = laplace_interval(mean, scale, 0.9)
        expected = -0.1 * math.log(0.1)
        self.assertAlmostEqual(float(upper - mean), expected, places=6)
        self.assertAlmostEqual(float(mean - lower), expected, places=6)

    def test_von_mises_interval_is_circular_and_monotone(self) -> None:
        lookup = VonMisesIntervalLookup.build(0.9, kappa_points=32, angle_points=1025)
        widths = lookup.half_width(np.array([0.0, 1.0, 10.0, 100.0]))
        self.assertAlmostEqual(float(widths[0]), 0.9 * math.pi, places=3)
        self.assertTrue(np.all(np.diff(widths) < 0.0))
        variance = circular_variance(torch.tensor([0.0, 1.0, 100.0]))
        self.assertTrue(torch.all(variance[:-1] > variance[1:]))

    def test_correlation_risk_and_coverage(self) -> None:
        uncertainty = np.arange(10, dtype=np.float64)
        error = uncertainty.copy()
        self.assertAlmostEqual(spearman_correlation(uncertainty, error), 1.0)
        curve = risk_coverage_curve(error, -uncertainty, (0.5, 1.0))
        self.assertLess(curve[0]["mean_error"], curve[1]["mean_error"])
        self.assertEqual(
            empirical_interval_coverage(
                np.array([0.1, 0.4]), np.array([0.0, 0.0]), np.array([0.2, 0.3])
            ),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
