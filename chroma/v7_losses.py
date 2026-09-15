"""Stable supervised and measurement-consistency losses for V7."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .chroma_polar import cartesian_chroma_to_polar
from .research_data import SITING_OFFSETS, DegradationSpec, _downsample_matrix
from .v7 import V7Prediction


@dataclass(frozen=True)
class V7LossConfig:
    lambda_cart: float = 1.0
    lambda_amplitude: float = 0.1
    lambda_phase: float = 0.05
    lambda_forward: float = 0.0
    phase_reference_amplitude: float = 0.05
    eps: float = 1e-6
    probabilistic: bool = True
    debug_finite: bool = False

    def validate(self) -> None:
        for name in (
            "lambda_cart",
            "lambda_amplitude",
            "lambda_phase",
            "lambda_forward",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if self.phase_reference_amplitude <= 0.0 or self.eps <= 0.0:
            raise ValueError("phase_reference_amplitude and eps must be positive")


def log_i0(kappa: torch.Tensor) -> torch.Tensor:
    """Stable `log(I0(kappa))`, evaluated in float32 under mixed precision."""
    values = kappa.float()
    if not hasattr(torch.special, "i0e"):
        raise RuntimeError("V7 requires torch.special.i0e (PyTorch >= 1.8)")
    result = (
        torch.log(torch.special.i0e(values).clamp_min(torch.finfo(values.dtype).tiny))
        + values.abs()
    )
    return result.to(kappa.dtype)


def laplace_nll(
    target: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    stable_scale = scale.clamp_min(eps)
    return (target - mean).abs() / stable_scale + torch.log(2.0 * stable_scale)


def von_mises_nll(
    target_phase: torch.Tensor,
    mean_phase: torch.Tensor,
    kappa: torch.Tensor,
) -> torch.Tensor:
    return (
        -kappa * torch.cos(target_phase - mean_phase)
        + math.log(2.0 * math.pi)
        + log_i0(kappa)
    )


def phase_weight(
    target_amplitude: torch.Tensor, reference_amplitude: float
) -> torch.Tensor:
    if reference_amplitude <= 0.0:
        raise ValueError("reference_amplitude must be positive")
    return (target_amplitude / reference_amplitude).clamp(0.0, 1.0)


def degrade_chroma_torch(chroma: torch.Tensor, spec: DegradationSpec) -> torch.Tensor:
    """Differentiable equivalent of publication `downsample_chroma`."""
    spec.validate()
    if chroma.ndim != 4 or chroma.shape[1] != 2:
        raise ValueError(f"Expected Bx2xHxW chroma, got {tuple(chroma.shape)}")
    height, width = chroma.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError("Forward consistency requires even spatial dimensions")
    offset_x, offset_y = SITING_OFFSETS[spec.siting]
    weights_x = torch.as_tensor(
        _downsample_matrix(width, offset_x, spec.downsample_filter),
        dtype=chroma.dtype,
        device=chroma.device,
    )
    weights_y = torch.as_tensor(
        _downsample_matrix(height, offset_y, spec.downsample_filter),
        dtype=chroma.dtype,
        device=chroma.device,
    )
    vertical = torch.einsum("ih,bchw->bciw", weights_y, chroma)
    return torch.einsum("jw,bciw->bcij", weights_x, vertical)


def forward_consistency_loss(
    predicted_chroma: torch.Tensor,
    observed_low: torch.Tensor,
    specs: DegradationSpec | Sequence[DegradationSpec],
) -> torch.Tensor:
    if isinstance(specs, DegradationSpec):
        degraded = degrade_chroma_torch(predicted_chroma, specs)
    else:
        if len(specs) != predicted_chroma.shape[0]:
            raise ValueError("One degradation spec is required per batch item")
        degraded = torch.cat(
            [
                degrade_chroma_torch(predicted_chroma[index : index + 1], spec)
                for index, spec in enumerate(specs)
            ],
            dim=0,
        )
    if degraded.shape != observed_low.shape:
        raise ValueError(
            f"Observed low chroma shape {observed_low.shape} != {degraded.shape}"
        )
    return F.l1_loss(degraded, observed_low)


def v7_supervised_loss(
    prediction: V7Prediction,
    target_ycrcb: torch.Tensor,
    neutral_chroma: float,
    config: V7LossConfig,
    observed_low: torch.Tensor | None = None,
    degradation_specs: DegradationSpec | Sequence[DegradationSpec] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    config.validate()
    target_amplitude, target_phase = cartesian_chroma_to_polar(
        target_ycrcb[:, 1:3], neutral_chroma
    )
    cartesian = F.l1_loss(prediction.chroma_mean, target_ycrcb[:, 1:3])
    amplitude_map = laplace_nll(
        target_amplitude,
        prediction.amplitude_mean,
        prediction.amplitude_scale,
        config.eps,
    )
    amplitude = amplitude_map.mean()
    weights = phase_weight(target_amplitude, config.phase_reference_amplitude)
    phase_map = von_mises_nll(
        target_phase, prediction.phase_mean, prediction.phase_kappa
    )
    phase = (weights * phase_map).sum() / weights.sum().clamp_min(config.eps)
    forward = cartesian.new_zeros(())
    if config.lambda_forward > 0.0:
        if observed_low is None or degradation_specs is None:
            raise ValueError(
                "Forward loss is enabled but observed low chroma/specs were not provided"
            )
        forward = forward_consistency_loss(
            prediction.chroma_mean, observed_low, degradation_specs
        )
    probabilistic = 1.0 if config.probabilistic else 0.0
    total = (
        config.lambda_cart * cartesian
        + probabilistic * config.lambda_amplitude * amplitude
        + probabilistic * config.lambda_phase * phase
        + config.lambda_forward * forward
    )
    components = {
        "cartesian_l1": cartesian,
        "amplitude_nll": amplitude,
        "phase_nll_weighted": phase,
        "forward_l1": forward,
        "phase_weight_mean": weights.mean(),
        "total": total,
    }
    if config.debug_finite:
        for name, value in components.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite V7 loss component: {name}")
    return total, components
