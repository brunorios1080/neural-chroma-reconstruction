"""Independent invariants for the publication comparison; run on a compute node."""
import unittest
import numpy as np

from chroma.li2026 import (MATRICES, cpsnr_mae, iround, lift_forward, lift_inverse,
    lifting_factors, model_input, model_output, reconstruct, scaled_transform, upsample_chroma)


class TestLi2026(unittest.TestCase):
    def test_psnr_units_and_rounding(self):
        a = np.full((12, 12, 3), 30.)
        metrics = cpsnr_mae(a, a + 1)
        self.assertAlmostEqual(metrics['cpsnr_rgb'], 48.1308036087)
        self.assertEqual(metrics['mae_rgb_255'], 1)
        np.testing.assert_array_equal(iround(np.array([-1.5, -.5, .5, 1.5])), [-2, -1, 1, 2])

    def test_integer_reversibility_and_factorization(self):
        rgb = np.random.default_rng(42).integers(0, 256, (10000, 3))
        for a0 in MATRICES.values():
            _, a = scaled_transform(a0)
            self.assertAlmostEqual(np.linalg.det(a), 1, places=10)
            factors = lifting_factors(a)
            np.testing.assert_allclose(factors[0] @ factors[1] @ factors[2], a, atol=1e-10)
            np.testing.assert_array_equal(lift_inverse(lift_forward(rgb, factors), factors), rgb)

    def test_float_scaling_cannot_improve_linear_interpolation(self):
        rgb = np.random.default_rng(12).uniform(30, 200, (24, 32, 3))
        for sampling in ('cosited_point', 'center_box'):
            for method in ('bilinear', 'bicubic'):
                conventional, _, _ = reconstruct(rgb, sampling, method, quantize=False)
                scaled, _, _ = reconstruct(rgb, sampling, method, transform='scaled_matrix', quantize=False)
                np.testing.assert_allclose(conventional, scaled, atol=1e-10, rtol=0)

    def test_sampling_coordinates_and_constant_preservation(self):
        yy, xx = np.indices((12, 16))
        low = np.stack((3*yy + 2*xx + 50, 2*yy + xx + 80), axis=-1)
        for method in ('bilinear', 'bicubic'):
            full = upsample_chroma(low, (24, 32), 0, method)
            np.testing.assert_array_equal(full[::2, ::2], low)
            y, x = np.indices((24, 32))
            expected = np.stack((1.5*y + x + 50, y + .5*x + 80), axis=-1)
            np.testing.assert_allclose(full[4:-4, 4:-4], expected[4:-4, 4:-4], atol=1e-10)
        rgb = np.full((23, 31, 3), [30., 100., 200.])
        for sampling in ('cosited_point', 'center_box'):
            result, _, _ = reconstruct(rgb, sampling, quantize=False)
            np.testing.assert_allclose(result, rgb, atol=1e-10)

    def test_retained_sites_and_model_range_adapter(self):
        rgb = np.random.default_rng(8).integers(0, 256, (32, 40, 3))
        for matrix in MATRICES:
            hybrid, _, _ = reconstruct(rgb, matrix=matrix, transform='scaled_hybrid')
            np.testing.assert_array_equal(hybrid[::2, ::2], rgb[::2, ::2])
            baseline, observation, _ = reconstruct(rgb, matrix=matrix)
            converted = model_output(model_input(observation, matrix), matrix)
            # The float32 model interface can only affect exact rounding ties.
            self.assertLessEqual(np.abs(converted - baseline).max(), 1.)
            self.assertLess(np.mean(np.abs(converted - baseline)), .01)

    def test_decode_first_only_uses_retained_chroma_and_luma(self):
        rgb = np.random.default_rng(81).integers(20, 230, (32, 40, 3)).astype(float)
        prediction, observation, _ = reconstruct(rgb, transform='scaled_decode_first')
        np.testing.assert_array_equal(prediction[::2, ::2], rgb[::2, ::2])
        self.assertTrue(np.isfinite(observation).all())
        changed = rgb.copy()
        y_row = MATRICES['paper_rounded'][0]
        changed[1::2, 1::2] += np.array([y_row[1], -y_row[0], 0.])
        prediction2, observation2, _ = reconstruct(changed, transform='scaled_decode_first')
        np.testing.assert_array_equal(observation, observation2)
        np.testing.assert_array_equal(prediction, prediction2)


if __name__ == '__main__':
    unittest.main()
