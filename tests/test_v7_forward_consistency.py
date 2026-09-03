import unittest

import numpy as np
import torch

from chroma.research_data import DegradationSpec, downsample_chroma
from chroma.v7_losses import degrade_chroma_torch, forward_consistency_loss


class ForwardConsistencyTests(unittest.TestCase):
    def test_torch_operator_exactly_matches_publication_operator(self) -> None:
        generator = np.random.default_rng(8)
        chroma = generator.random((10, 12, 2), dtype=np.float32)
        tensor = torch.from_numpy(chroma).permute(2, 0, 1).unsqueeze(0)
        for siting in ("center", "left", "cosited"):
            for filter_name in ("box", "triangle", "gaussian", "lanczos3", "point"):
                spec = DegradationSpec("test", siting, filter_name, "bilinear")
                expected = downsample_chroma(chroma, siting, filter_name)
                actual = degrade_chroma_torch(tensor, spec)[0].permute(1, 2, 0).numpy()
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)

    def test_forward_loss_is_zero_for_observation_and_differentiable(self) -> None:
        chroma = torch.rand(2, 2, 8, 10, requires_grad=True)
        spec = DegradationSpec("center_box", "center", "box", "bilinear")
        observed = degrade_chroma_torch(chroma.detach(), spec)
        loss = forward_consistency_loss(chroma, observed, spec)
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=7)
        loss.backward()
        self.assertIsNotNone(chroma.grad)


if __name__ == "__main__":
    unittest.main()
