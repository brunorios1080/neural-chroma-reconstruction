import math
import unittest

import torch

from chroma.chroma_polar import circular_difference
from chroma.v7 import V7Config, V7PolarChromaRefiner
from chroma.v7_losses import (
    V7LossConfig,
    laplace_nll,
    phase_weight,
    v7_supervised_loss,
    von_mises_nll,
)


class V7LossTests(unittest.TestCase):
    def test_losses_are_finite_and_backward(self) -> None:
        model = V7PolarChromaRefiner(V7Config(width=8, depth=1))
        inputs = torch.rand(2, 3, 8, 8)
        target = torch.rand(2, 3, 8, 8)
        total, components = v7_supervised_loss(
            model(inputs), target, 0.5, V7LossConfig(debug_finite=True)
        )
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(
            set(components),
            {
                "total",
                "cartesian_l1",
                "amplitude_nll",
                "phase_nll_weighted",
                "forward_l1",
                "phase_weight_mean",
            },
        )
        total.backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )

    def test_neutral_phase_is_zero_weight(self) -> None:
        amplitudes = torch.tensor([0.0, 0.025, 0.05, 0.2])
        torch.testing.assert_close(
            phase_weight(amplitudes, 0.05), torch.tensor([0.0, 0.5, 1.0, 1.0])
        )
        model = V7PolarChromaRefiner(V7Config(width=8, depth=1))
        inputs = torch.full((1, 3, 8, 8), 0.5)
        inputs[:, 0] = 0.25
        total, components = v7_supervised_loss(
            model(inputs), inputs, 0.5, V7LossConfig(debug_finite=True)
        )
        self.assertEqual(float(components["phase_weight_mean"]), 0.0)
        self.assertEqual(float(components["phase_nll_weighted"].detach()), 0.0)
        self.assertTrue(torch.isfinite(total))

    def test_hue_wrap_has_small_circular_error_and_likelihood(self) -> None:
        target = torch.tensor([math.pi - 1e-4])
        mean = torch.tensor([-math.pi + 1e-4])
        self.assertLess(float(circular_difference(target, mean).abs()), 3e-4)
        close = von_mises_nll(target, mean, torch.tensor([10.0]))
        far = von_mises_nll(target, torch.tensor([0.0]), torch.tensor([10.0]))
        self.assertLess(float(close), float(far))

    def test_distribution_extremes_are_finite(self) -> None:
        values = laplace_nll(
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.5, 0.5]),
            torch.tensor([1e-30, 1e6]),
        )
        circular = von_mises_nll(
            torch.tensor([-math.pi, 0.0]),
            torch.tensor([math.pi, math.pi]),
            torch.tensor([1e-4, 100.0]),
        )
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.isfinite(circular).all())

    def test_tiny_batch_can_overfit_cartesian_target(self) -> None:
        torch.manual_seed(4)
        model = V7PolarChromaRefiner(V7Config(width=12, depth=2))
        inputs = torch.rand(1, 3, 8, 8) * 0.6 + 0.2
        target = inputs.clone()
        target[:, 1] = (target[:, 1] + 0.04).clamp(0.0, 1.0)
        target[:, 2] = (target[:, 2] - 0.03).clamp(0.0, 1.0)
        config = V7LossConfig(
            lambda_cart=1.0,
            lambda_amplitude=0.0,
            lambda_phase=0.0,
            probabilistic=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        initial = None
        for _ in range(100):
            prediction = model(inputs)
            loss, _ = v7_supervised_loss(prediction, target, 0.5, config)
            if initial is None:
                initial = float(loss.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss.detach()), initial * 0.25)


if __name__ == "__main__":
    unittest.main()
