"""Full-reference quality metrics used by the expanded research benchmark."""

from __future__ import annotations

import math

import numpy as np

from .research_data import ycrcb_to_rgb


def psnr(reference: np.ndarray, candidate: np.ndarray, cap: float = 80.0) -> float:
    error = float(
        np.mean(
            (reference.astype(np.float64) - candidate.astype(np.float64)) ** 2
        )
    )
    if error <= 1e-15:
        return cap
    return min(cap, 10.0 * math.log10(1.0 / error))


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    coordinates = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    kernel = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def _convolve_axis(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    padding = [(0, 0)] * image.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(image, padding, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=len(kernel), axis=axis
    )
    return np.tensordot(windows, kernel, axes=([-1], [0]))


def _gaussian_blur(image: np.ndarray, size: int = 11, sigma: float = 1.5) -> np.ndarray:
    kernel = _gaussian_kernel(size, sigma)
    return _convolve_axis(_convolve_axis(image, kernel, 0), kernel, 1)


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean SSIM with an 11x11 Gaussian window and unit data range."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError("SSIM inputs must have identical shapes")
    if reference.ndim == 2:
        reference = reference[..., None]
        candidate = candidate[..., None]
    window = min(11, reference.shape[0], reference.shape[1])
    if window % 2 == 0:
        window -= 1
    if window < 3:
        raise ValueError("SSIM requires dimensions of at least 3 pixels")
    sigma = 1.5 * window / 11.0
    mu_x = _gaussian_blur(reference, window, sigma)
    mu_y = _gaussian_blur(candidate, window, sigma)
    mu_x_sq, mu_y_sq, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_x_sq = _gaussian_blur(reference * reference, window, sigma) - mu_x_sq
    sigma_y_sq = _gaussian_blur(candidate * candidate, window, sigma) - mu_y_sq
    sigma_xy = _gaussian_blur(reference * candidate, window, sigma) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (
        sigma_x_sq + sigma_y_sq + c2
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-15)))


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert normalized sRGB to CIE Lab under the D65 reference white."""
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    matrix = np.asarray(
        (
            (0.4124564, 0.3575761, 0.1804375),
            (0.2126729, 0.7151522, 0.0721750),
            (0.0193339, 0.1191920, 0.9503041),
        ),
        dtype=np.float64,
    )
    xyz = linear @ matrix.T
    xyz /= np.asarray((0.95047, 1.0, 1.08883), dtype=np.float64)
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    lightness = 116.0 * transformed[..., 1] - 16.0
    a_star = 500.0 * (transformed[..., 0] - transformed[..., 1])
    b_star = 200.0 * (transformed[..., 1] - transformed[..., 2])
    return np.stack((lightness, a_star, b_star), axis=-1)


def delta_e_ciede2000_lab(lab_1: np.ndarray, lab_2: np.ndarray) -> np.ndarray:
    """Vectorized CIEDE2000 implementation following Sharma et al. (2005)."""
    lab_1 = np.asarray(lab_1, dtype=np.float64)
    lab_2 = np.asarray(lab_2, dtype=np.float64)
    if lab_1.shape != lab_2.shape or lab_1.shape[-1] != 3:
        raise ValueError("CIEDE2000 inputs must have matching [..., 3] shapes")
    l_1, a_1, b_1 = np.moveaxis(lab_1, -1, 0)
    l_2, a_2, b_2 = np.moveaxis(lab_2, -1, 0)
    c_1 = np.hypot(a_1, b_1)
    c_2 = np.hypot(a_2, b_2)
    c_bar = (c_1 + c_2) / 2.0
    c_bar_7 = c_bar**7
    g = 0.5 * (1.0 - np.sqrt(c_bar_7 / (c_bar_7 + 25.0**7)))
    a_1_prime = (1.0 + g) * a_1
    a_2_prime = (1.0 + g) * a_2
    c_1_prime = np.hypot(a_1_prime, b_1)
    c_2_prime = np.hypot(a_2_prime, b_2)

    def hue(a_value: np.ndarray, b_value: np.ndarray) -> np.ndarray:
        angle = np.degrees(np.arctan2(b_value, a_value))
        return np.mod(angle, 360.0)

    h_1_prime = hue(a_1_prime, b_1)
    h_2_prime = hue(a_2_prime, b_2)
    delta_l = l_2 - l_1
    delta_c = c_2_prime - c_1_prime
    hue_difference = h_2_prime - h_1_prime
    zero_chroma = (c_1_prime * c_2_prime) <= 1e-15
    hue_difference = np.where(zero_chroma, 0.0, hue_difference)
    hue_difference = np.where(hue_difference > 180.0, hue_difference - 360.0, hue_difference)
    hue_difference = np.where(hue_difference < -180.0, hue_difference + 360.0, hue_difference)
    delta_h = 2.0 * np.sqrt(c_1_prime * c_2_prime) * np.sin(
        np.radians(hue_difference / 2.0)
    )
    l_bar = (l_1 + l_2) / 2.0
    c_bar_prime = (c_1_prime + c_2_prime) / 2.0
    hue_sum = h_1_prime + h_2_prime
    hue_bar = np.where(zero_chroma, hue_sum, hue_sum / 2.0)
    hue_bar = np.where(
        (~zero_chroma) & (np.abs(h_1_prime - h_2_prime) > 180.0) & (hue_sum < 360.0),
        (hue_sum + 360.0) / 2.0,
        hue_bar,
    )
    hue_bar = np.where(
        (~zero_chroma) & (np.abs(h_1_prime - h_2_prime) > 180.0) & (hue_sum >= 360.0),
        (hue_sum - 360.0) / 2.0,
        hue_bar,
    )
    t = (
        1.0
        - 0.17 * np.cos(np.radians(hue_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hue_bar))
        + 0.32 * np.cos(np.radians(3.0 * hue_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hue_bar - 63.0))
    )
    delta_theta = 30.0 * np.exp(-((hue_bar - 275.0) / 25.0) ** 2)
    r_c = 2.0 * np.sqrt(c_bar_prime**7 / (c_bar_prime**7 + 25.0**7))
    s_l = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / np.sqrt(20.0 + (l_bar - 50.0) ** 2)
    s_c = 1.0 + 0.045 * c_bar_prime
    s_h = 1.0 + 0.015 * c_bar_prime * t
    r_t = -np.sin(np.radians(2.0 * delta_theta)) * r_c
    l_term = delta_l / s_l
    c_term = delta_c / s_c
    h_term = delta_h / s_h
    return np.sqrt(
        np.maximum(
            0.0,
            l_term**2 + c_term**2 + h_term**2 + r_t * c_term * h_term,
        )
    )


def delta_e_ciede2000(reference_rgb: np.ndarray, candidate_rgb: np.ndarray) -> np.ndarray:
    return delta_e_ciede2000_lab(
        srgb_to_lab(reference_rgb), srgb_to_lab(candidate_rgb)
    )


def _chroma_gradient(chroma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dy = np.gradient(chroma.astype(np.float64), axis=0)
    dx = np.gradient(chroma.astype(np.float64), axis=1)
    return dx, dy


def reconstruction_metrics(
    reference_ycrcb: np.ndarray,
    candidate_ycrcb: np.ndarray,
    psnr_cap: float = 80.0,
) -> dict[str, float]:
    reference = np.clip(np.asarray(reference_ycrcb, dtype=np.float32), 0.0, 1.0)
    candidate = np.clip(np.asarray(candidate_ycrcb, dtype=np.float32), 0.0, 1.0)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Metric inputs must match, got {reference.shape} and {candidate.shape}"
        )
    reference_rgb = ycrcb_to_rgb(reference)
    candidate_rgb = ycrcb_to_rgb(candidate)
    reference_chroma = reference[..., 1:3]
    candidate_chroma = candidate[..., 1:3]
    absolute_chroma = np.mean(np.abs(reference_chroma - candidate_chroma), axis=2)
    reference_dx, reference_dy = _chroma_gradient(reference_chroma)
    candidate_dx, candidate_dy = _chroma_gradient(candidate_chroma)
    edge_strength = np.mean(np.hypot(reference_dx, reference_dy), axis=2)
    threshold = np.percentile(edge_strength, 75.0)
    edge_mask = edge_strength >= threshold
    gradient_error = 0.5 * (
        np.mean(np.abs(reference_dx - candidate_dx))
        + np.mean(np.abs(reference_dy - candidate_dy))
    )
    delta_e = delta_e_ciede2000(reference_rgb, candidate_rgb)
    return {
        "rgb_psnr": psnr(reference_rgb, candidate_rgb, psnr_cap),
        "chroma_psnr": psnr(reference_chroma, candidate_chroma, psnr_cap),
        "rgb_ssim": ssim(reference_rgb, candidate_rgb),
        "chroma_ssim": ssim(reference_chroma, candidate_chroma),
        "delta_e2000_mean": float(np.mean(delta_e)),
        "delta_e2000_p95": float(np.percentile(delta_e, 95.0)),
        "chroma_mae": float(np.mean(absolute_chroma)),
        "chroma_edge_mae": float(np.mean(absolute_chroma[edge_mask])),
        "chroma_gradient_mae": float(gradient_error),
        "boundary_strength": float(np.mean(edge_strength[edge_mask])),
    }
