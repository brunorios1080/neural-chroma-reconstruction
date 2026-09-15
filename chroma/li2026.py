"""Auditable reconstruction of Li et al., IET Image Processing e70338 (2026).

This is an independent implementation, not author-supplied code. Undocumented
sampling, boundaries, and the transition between lifting and matrix transforms
are explicit experiment settings. Published tables are not treated as reproduced
unless the conventional baselines and transform checks agree.
"""
from __future__ import annotations

import math
import numpy as np

PAPER_MATRIX = np.array([[.2568, .5041, .0979], [-.1482, -.2910, .4392],
                         [.4392, -.3678, -.0714]], dtype=np.float64)
BT601_MATRIX = np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112],
                         [112, -93.786, -18.214]], dtype=np.float64) / 255
FULL_MATRIX = np.array([[.299, .587, .114], [-.168736, -.331264, .5],
                        [.5, -.418688, -.081312]], dtype=np.float64)
MATRICES = {'paper_rounded': PAPER_MATRIX, 'bt601_precise': BT601_MATRIX,
            'bt601_full': FULL_MATRIX}


def iround(x):
    """Round ties away from zero (MATLAB-style); never clip signed chroma."""
    return np.copysign(np.floor(np.abs(x) + .5), x)


def scaled_transform(a0=PAPER_MATRIX):
    """Equations 10–17. The multiplier's scale does not change the root F."""
    d = 1 / abs(np.linalg.det(a0))
    omega = 1 / (2 * abs(np.linalg.det(a0)) * (a0 * a0).sum(axis=1))
    def sigma(lam):
        return (1 + np.sqrt(1 + 4 * lam * omega)) / 2
    low, high = 0., 1.
    while np.prod(sigma(high)) < d:
        high *= 2
    for _ in range(100):
        middle = (low + high) / 2
        if np.prod(sigma(middle)) < d:
            low = middle
        else:
            high = middle
    scale = sigma((low + high) / 2)
    return scale, scale[:, None] * a0


def lifting_factors(a, literal_equation27=False):
    """Factorize A, explicitly auditing the sign in the rendered Eq. 27.

    The online equation prints +M31 in l5. Matrix multiplication instead requires
    -M31 with Eq. 21's definition. The literal variant is exposed for the audit;
    it must never be silently represented as a valid factorization of A.
    """
    a11, a12, a13 = a[0]
    a21, a22, a23 = a[1]
    a31, _, a33 = a[2]
    m22 = a11 * a33 - a13 * a31
    m31 = a12 * a23 - a13 * a22
    m32 = a11 * a23 - a13 * a21
    l1, l2, l3 = (a23 - m32) / a13, (a33 - m22) / a13, (m22 - 1) / m32
    l4 = (a11 - 1) / a13
    l5 = (a12 * m32 - a13 + (m31 if literal_equation27 else -m31)) / (a13 * m32)
    u1, u2, u3 = (a13 + m31) / m32, a13, m32
    lower = np.array([[1, 0, 0], [l1, 1, 0], [l2, l3, 1.]])
    upper = np.array([[1, u1, u2], [0, 1, u3], [0, 0, 1.]])
    first = np.array([[1., 0, 0], [0, 1, 0], [l4, l5, 1]])
    if not literal_equation27 and not np.allclose(lower @ upper @ first, a, atol=1e-10, rtol=0):
        raise ValueError('Published factorization does not reconstruct A')
    return lower, upper, first


def lift_forward(rgb, factors):
    value = np.asarray(rgb, dtype=np.float64)
    # In each forward equation the RHS refers to the stage INPUT. Updating that
    # input in place instead would implement a different linear transform.
    for factor in reversed(factors):
        value = value + iround(value @ (factor - np.eye(3)).T)
    return value


def lift_inverse(ycc, factors):
    value = np.array(ycc, dtype=np.float64, copy=True)
    for factor in factors:
        indices = range(3) if np.allclose(factor, np.tril(factor)) else range(2, -1, -1)
        for i in indices:
            row = factor[i].copy()
            row[i] = 0
            value[..., i] -= iround(value @ row)
    return value


def cubic_kernel(distance):
    x = np.abs(distance)
    return np.where(x <= 1, 1.5 * x ** 3 - 2.5 * x ** 2 + 1,
                    np.where(x < 2, -.5 * x ** 3 + 2.5 * x ** 2 - 4 * x + 2, 0))


def resample_axis(value, length, offset, axis, method):
    """Sample spacing 2; edge replication; Keys cubic uses a=-0.5 (Eq. 9)."""
    positions = (np.arange(length, dtype=np.float64) - offset) / 2
    start = np.floor(positions).astype(int)
    result = None
    for shift in ((0, 1) if method == 'bilinear' else (-1, 0, 1, 2)):
        index = start + shift
        distance = positions - index
        weight = np.maximum(0., 1 - np.abs(distance)) if method == 'bilinear' else cubic_kernel(distance)
        shape = [1] * value.ndim
        shape[axis] = length
        term = np.take(value, np.clip(index, 0, value.shape[axis] - 1), axis=axis) * weight.reshape(shape)
        result = term if result is None else result + term
    return result


def sample_chroma(ycc, sampling):
    if sampling == 'cosited_point':
        return ycc[::2, ::2, 1:].copy(), 0.
    if sampling == 'center_box':
        h, w = ycc.shape[:2]
        padded = np.pad(ycc[..., 1:], ((0, h % 2), (0, w % 2), (0, 0)), mode='edge')
        return (padded[::2, ::2] + padded[1::2, ::2] + padded[::2, 1::2] + padded[1::2, 1::2]) / 4, .5
    raise ValueError(sampling)


def upsample_chroma(low, shape, offset, method):
    value = resample_axis(low, shape[0], offset, 0, method)
    return resample_axis(value, shape[1], offset, 1, method)


def reconstruct(rgb, sampling='cosited_point', method='bilinear', matrix='paper_rounded',
                transform='conventional', quantize=True):
    """Return reconstructed RGB and the actual upsampled observation.

    Offsets (16,128,128) cancel; calculations use signed, offset-free values.
    scaled_hybrid uses lifting at retained sites, matrix mapping elsewhere as
    described after Eq. 43. scaled_lifting uses lifting for every forward pixel.
    Both interpretations are retained because the paper gives no executable code.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    a0 = MATRICES[matrix]
    scale, a = scaled_transform(a0) if transform != 'conventional' else (np.ones(3), a0)
    ycc = rgb @ a.T
    if quantize:
        ycc = iround(ycc)
    factors = None
    if transform in ('scaled_hybrid', 'scaled_lifting', 'scaled_decode_first'):
        if sampling != 'cosited_point' or not quantize:
            raise ValueError('Retained-site lifting requires point samples and integer transforms')
        factors = lifting_factors(a)
        if transform == 'scaled_lifting':
            ycc = lift_forward(rgb, factors)
        else:
            ycc[::2, ::2] = lift_forward(rgb[::2, ::2], factors)
    low, offset = sample_chroma(ycc, sampling)
    if transform == 'scaled_decode_first':
        # A decoder can recover the retained RGB triplets from their lifting
        # codes, then obtain unrounded chroma for interpolation. This uses only
        # transmitted samples, never the unavailable original off-grid chroma.
        retained_rgb = lift_inverse(ycc[::2, ::2], factors)
        low = (retained_rgb @ a.T)[..., 1:]
    observation = ycc.copy()
    observation[..., 1:] = upsample_chroma(low, rgb.shape[:2], offset, method)
    reconstructed = observation @ np.linalg.inv(a).T
    if quantize:
        reconstructed = iround(reconstructed)
    if factors is not None:
        reconstructed[::2, ::2] = lift_inverse(ycc[::2, ::2], factors)
    return np.clip(reconstructed, 0, 255), observation, scale


def cpsnr_mae(reference, candidate):
    difference = reference.astype(np.float64) - candidate.astype(np.float64)
    mse = float(np.mean(difference * difference))
    return {'cpsnr_rgb': 10 * math.log10(255 ** 2 / mse) if mse else float('inf'),
            'mae_rgb_255': float(np.mean(np.abs(difference))), 'mse_rgb_255': mse}


def model_input(observation, matrix='paper_rounded', neutral=.5):
    """Convert the *observed* signed YCbCr to models' normalized YCrCb."""
    a0 = MATRICES[matrix]
    ranges = np.array([a0[0].sum(), a0[2, 0] * 2, a0[1, 2] * 2]) * 255
    value = observation[..., [0, 2, 1]] / ranges
    value[..., 1:] += neutral
    return value.astype(np.float32)


def model_output(prediction, matrix='paper_rounded', neutral=.5):
    a0 = MATRICES[matrix]
    value = prediction.astype(np.float64).copy()
    value[..., 1:] -= neutral
    ranges = np.array([a0[0].sum(), a0[2, 0] * 2, a0[1, 2] * 2]) * 255
    ycc = (value * ranges)[..., [0, 2, 1]]
    return np.clip(iround(ycc @ np.linalg.inv(a0).T), 0, 255)
