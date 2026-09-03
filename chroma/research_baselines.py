"""Classical chroma upsampling baselines and learned-model adapters."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .checkpoints import load_model
from .models import build_model
from .research_data import SITING_OFFSETS, downsample_chroma, upsample_chroma
from .research_models import load_ablation_checkpoint
from .v7 import load_v7_checkpoint

CLASSICAL_METHODS = (
    "nearest",
    "bilinear",
    "bicubic",
    "lanczos3",
    "guided",
    "joint_bilateral",
)


def _moving_mean_axis(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    if radius <= 0:
        return values.astype(np.float64, copy=True)
    width = 2 * radius + 1
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="edge")
    cumulative = np.cumsum(padded, axis=axis, dtype=np.float64)
    zero_shape = list(cumulative.shape)
    zero_shape[axis] = 1
    cumulative = np.concatenate((np.zeros(zero_shape), cumulative), axis=axis)
    high = [slice(None)] * values.ndim
    low = [slice(None)] * values.ndim
    high[axis] = slice(width, width + values.shape[axis])
    low[axis] = slice(0, values.shape[axis])
    return (cumulative[tuple(high)] - cumulative[tuple(low)]) / width


def box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    return _moving_mean_axis(_moving_mean_axis(values, radius, 0), radius, 1)


def guided_filter_chroma(
    luma: np.ndarray,
    initial_chroma: np.ndarray,
    radius: int = 4,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """Guided filtering with full-resolution luma as the guide image."""
    guide = np.asarray(luma, dtype=np.float64)
    chroma = np.asarray(initial_chroma, dtype=np.float64)
    mean_guide = box_mean(guide, radius)
    variance = box_mean(guide * guide, radius) - mean_guide * mean_guide
    outputs = []
    for channel in range(2):
        plane = chroma[..., channel]
        mean_plane = box_mean(plane, radius)
        covariance = box_mean(guide * plane, radius) - mean_guide * mean_plane
        coefficient_a = covariance / (variance + epsilon)
        coefficient_b = mean_plane - coefficient_a * mean_guide
        output = box_mean(coefficient_a, radius) * guide + box_mean(
            coefficient_b, radius
        )
        outputs.append(output)
    return np.clip(np.stack(outputs, axis=2), 0.0, 1.0).astype(np.float32)


def joint_bilateral_upsample(
    luma: np.ndarray,
    low_chroma: np.ndarray,
    siting: str,
    radius: int = 2,
    spatial_sigma: float = 2.0,
    range_sigma: float = 0.08,
) -> np.ndarray:
    """Joint bilateral upsampling in the low-resolution chroma neighborhood."""
    if siting not in SITING_OFFSETS:
        raise ValueError(f"Unsupported siting: {siting}")
    height, width = luma.shape
    low_luma = downsample_chroma(
        np.repeat(luma[..., None], 2, axis=2), siting, "box"
    )[..., 0]
    offset_x, offset_y = SITING_OFFSETS[siting]
    full_y, full_x = np.mgrid[:height, :width]
    low_positions_x = 2.0 * np.arange(low_chroma.shape[1]) + offset_x
    low_positions_y = 2.0 * np.arange(low_chroma.shape[0]) + offset_y
    base_x = np.argmin(np.abs(full_x[..., None] - low_positions_x), axis=2)
    base_y = np.argmin(np.abs(full_y[..., None] - low_positions_y), axis=2)
    numerator = np.zeros((height, width, 2), dtype=np.float64)
    denominator = np.zeros((height, width), dtype=np.float64)
    for delta_y in range(-radius, radius + 1):
        sample_y = np.clip(base_y + delta_y, 0, low_chroma.shape[0] - 1)
        position_y = 2.0 * sample_y + offset_y
        for delta_x in range(-radius, radius + 1):
            sample_x = np.clip(base_x + delta_x, 0, low_chroma.shape[1] - 1)
            position_x = 2.0 * sample_x + offset_x
            spatial_distance = (full_x - position_x) ** 2 + (full_y - position_y) ** 2
            sampled_luma = low_luma[sample_y, sample_x]
            range_distance = (luma - sampled_luma) ** 2
            weights = np.exp(
                -spatial_distance / (2.0 * spatial_sigma**2)
                - range_distance / (2.0 * range_sigma**2)
            )
            numerator += weights[..., None] * low_chroma[sample_y, sample_x]
            denominator += weights
    return np.clip(
        numerator / np.maximum(denominator[..., None], 1e-12), 0.0, 1.0
    ).astype(np.float32)


def classical_reconstruction(
    method: str,
    luma: np.ndarray,
    low_chroma: np.ndarray,
    siting: str,
) -> np.ndarray:
    if method in {"nearest", "bilinear", "bicubic", "lanczos3"}:
        return upsample_chroma(low_chroma, luma.shape, siting, method)
    bilinear = upsample_chroma(low_chroma, luma.shape, siting, "bilinear")
    if method == "guided":
        return guided_filter_chroma(luma, bilinear)
    if method == "joint_bilateral":
        return joint_bilateral_upsample(luma, low_chroma, siting)
    raise ValueError(f"Unknown classical baseline: {method}")


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(image))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )


def tensor_prediction(
    model: nn.Module,
    model_input: np.ndarray,
    device: torch.device,
    pad_multiple: int | None = None,
) -> np.ndarray:
    tensor = to_tensor(model_input, device)
    height, width = tensor.shape[-2:]
    if pad_multiple:
        pad_height = (-height) % pad_multiple
        pad_width = (-width) % pad_multiple
        if pad_height or pad_width:
            tensor = F.pad(tensor, (0, pad_width, 0, pad_height), mode="replicate")
    with torch.inference_mode():
        output = model(tensor)[0, :, :height, :width]
    return output.detach().float().cpu().permute(1, 2, 0).numpy()


def load_legacy_predictor(
    version: str,
    weights: str | Path,
    device: torch.device,
) -> tuple[nn.Module, Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    model = build_model(version).to(device)
    load_model(model, weights, version, device)
    model.eval()
    pad = 8 if version == "v5" else None
    predictor = lambda image: tensor_prediction(model, image, device, pad)
    return model, predictor, {"family": version, "weights": str(weights)}


def load_ablation_predictor(
    weights: str | Path,
    device: torch.device,
) -> tuple[nn.Module, Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    model, checkpoint = load_ablation_checkpoint(weights, device)
    model.eval()
    predictor = lambda image: tensor_prediction(model, image, device)
    return model, predictor, {
        "family": "ablation",
        "weights": str(weights),
        "config": checkpoint["config"],
        "epoch": checkpoint.get("epoch"),
    }


def load_v7_predictor(
    weights: str | Path,
    device: torch.device,
    mode: str = "mean",
) -> tuple[nn.Module, Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Load a strict V7 checkpoint for the common publication benchmark."""
    if mode not in {"mean", "safe"}:
        raise ValueError("V7 benchmark mode must be 'mean' or 'safe'")
    model, checkpoint = load_v7_checkpoint(weights, device)
    model.eval()

    def predictor(image: np.ndarray) -> np.ndarray:
        tensor = to_tensor(image, device)
        with torch.inference_mode():
            output = model(tensor).ycrcb(mode)[0]
        return output.detach().float().cpu().permute(1, 2, 0).numpy()

    return model, predictor, {
        "family": "v7",
        "mode": mode,
        "weights": str(weights),
        "architecture": checkpoint["architecture"],
        "loss": checkpoint["loss"],
        "epoch": checkpoint.get("epoch"),
    }


def load_torchscript_predictor(
    weights: str | Path,
    device: torch.device,
) -> tuple[nn.Module, Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    model = torch.jit.load(str(weights), map_location=device)
    model.eval()
    predictor = lambda image: tensor_prediction(model, image, device)
    return model, predictor, {"family": "torchscript", "weights": str(weights)}
