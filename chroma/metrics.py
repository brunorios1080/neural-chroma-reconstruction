"""Dependency-light PSNR and SSIM metrics for RGB and chroma comparisons."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .data import ycrcb_to_bgr_uint8


def psnr(reference: np.ndarray, candidate: np.ndarray, cap: float = 80.0) -> float:
    error = float(
        np.mean((reference.astype(np.float64) - candidate.astype(np.float64)) ** 2)
    )
    if error <= 1e-15:
        return cap
    return min(cap, 10.0 * math.log10(1.0 / error))


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute mean SSIM with the standard 11x11 Gaussian window."""
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    if reference.ndim == 2:
        reference = reference[:, :, None]
        candidate = candidate[:, :, None]

    height, width = reference.shape[:2]
    window = min(
        11, height if height % 2 else height - 1, width if width % 2 else width - 1
    )
    window = max(3, window)
    sigma = 1.5 * window / 11.0
    c1, c2 = 0.01**2, 0.03**2
    channel_scores = []
    for channel in range(reference.shape[2]):
        x = reference[:, :, channel]
        y = candidate[:, :, channel]
        mu_x = cv2.GaussianBlur(x, (window, window), sigma)
        mu_y = cv2.GaussianBlur(y, (window, window), sigma)
        mu_x_sq, mu_y_sq, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
        sigma_x_sq = cv2.GaussianBlur(x * x, (window, window), sigma) - mu_x_sq
        sigma_y_sq = cv2.GaussianBlur(y * y, (window, window), sigma) - mu_y_sq
        sigma_xy = cv2.GaussianBlur(x * y, (window, window), sigma) - mu_xy
        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        channel_scores.append(float(np.mean(numerator / denominator)))
    return float(np.mean(channel_scores))


def ycrcb_to_rgb_float(image: np.ndarray) -> np.ndarray:
    bgr = ycrcb_to_bgr_uint8(image)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def image_metrics(
    target_ycrcb: np.ndarray,
    candidate_ycrcb: np.ndarray,
    psnr_cap: float = 80.0,
) -> dict[str, float]:
    target = np.clip(target_ycrcb, 0.0, 1.0)
    candidate = np.clip(candidate_ycrcb, 0.0, 1.0)
    target_rgb = ycrcb_to_rgb_float(target)
    candidate_rgb = ycrcb_to_rgb_float(candidate)
    return {
        "rgb_psnr": psnr(target_rgb, candidate_rgb, psnr_cap),
        "rgb_ssim": ssim(target_rgb, candidate_rgb),
        "chroma_psnr": psnr(target[:, :, 1:3], candidate[:, :, 1:3], psnr_cap),
        "chroma_ssim": ssim(target[:, :, 1:3], candidate[:, :, 1:3]),
    }


def chroma_gradient_mae(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean absolute error between horizontal and vertical chroma gradients."""
    errors = []
    for channel in range(2):
        ref = reference[:, :, channel]
        pred = candidate[:, :, channel]
        for dx, dy in ((1, 0), (0, 1)):
            ref_gradient = cv2.Sobel(ref, cv2.CV_32F, dx, dy, ksize=3) / 8.0
            pred_gradient = cv2.Sobel(pred, cv2.CV_32F, dx, dy, ksize=3) / 8.0
            errors.append(np.abs(ref_gradient - pred_gradient))
    return float(np.mean(errors))


def reconstruction_metrics(
    reference_ycrcb: np.ndarray, candidate_ycrcb: np.ndarray
) -> dict[str, float]:
    """Image metrics plus chroma pixel, edge, and gradient errors."""
    reference_ycrcb = np.clip(reference_ycrcb, 0.0, 1.0)
    candidate_ycrcb = np.clip(candidate_ycrcb, 0.0, 1.0)
    scores = image_metrics(reference_ycrcb, candidate_ycrcb)
    reference_chroma = reference_ycrcb[:, :, 1:3]
    candidate_chroma = candidate_ycrcb[:, :, 1:3]
    chroma_error = np.mean(np.abs(reference_chroma - candidate_chroma), axis=2)

    edge_strength = np.zeros(reference_ycrcb.shape[:2], dtype=np.float32)
    for channel in range(2):
        chroma = reference_chroma[:, :, channel]
        dx = cv2.Sobel(chroma, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        dy = cv2.Sobel(chroma, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        edge_strength += np.hypot(dx, dy)
    edge_mask = edge_strength >= np.percentile(edge_strength, 75.0)

    scores.update(
        {
            "chroma_mae": float(np.mean(chroma_error)),
            "chroma_edge_mae": float(np.mean(chroma_error[edge_mask])),
            "chroma_gradient_mae": chroma_gradient_mae(
                reference_chroma, candidate_chroma
            ),
        }
    )
    return scores
