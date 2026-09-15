"""V7 recoverability-aware probabilistic polar chroma model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .chroma_polar import (
    PUBLICATION_NEUTRAL_CHROMA,
    cartesian_chroma_to_polar,
    circular_difference,
    maximum_amplitude_for_phase,
    polar_chroma_to_cartesian,
    wrap_angle,
)
from .models import ResBlock


def inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("inverse_softplus requires a positive value")
    return math.log(math.expm1(value))


@dataclass(frozen=True)
class V7Config:
    in_channels: int = 3
    width: int = 64
    depth: int = 8
    neutral_chroma: float = PUBLICATION_NEUTRAL_CHROMA
    eps: float = 1e-6
    kappa_min: float = 1e-4
    kappa_max: float = 100.0
    initial_amplitude_scale: float = 0.05
    initial_phase_concentration: float = 1.0
    safe_amplitude_scale_reference: float = 0.05
    safe_phase_kappa_reference: float = 2.0
    safe_phase_amplitude_reference: float = 0.05

    def validate(self) -> None:
        if self.in_channels != 3:
            raise ValueError("V7 currently requires [Y, Cr, Cb] three-channel input")
        if self.width < 4 or self.depth < 1:
            raise ValueError("V7 width must be >=4 and depth must be positive")
        if not 0.0 <= self.neutral_chroma <= 1.0:
            raise ValueError("neutral_chroma must be in [0,1]")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if not 0.0 < self.kappa_min < self.kappa_max:
            raise ValueError("Require 0 < kappa_min < kappa_max")
        for name in (
            "initial_amplitude_scale",
            "initial_phase_concentration",
            "safe_amplitude_scale_reference",
            "safe_phase_kappa_reference",
            "safe_phase_amplitude_reference",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass
class V7Prediction:
    neutral_chroma: float
    luma: torch.Tensor
    chroma_mean: torch.Tensor
    chroma_safe: torch.Tensor
    amplitude_baseline: torch.Tensor
    phase_baseline: torch.Tensor
    amplitude_mean: torch.Tensor
    phase_mean: torch.Tensor
    amplitude_scale: torch.Tensor
    phase_kappa: torch.Tensor
    amplitude_confidence: torch.Tensor
    phase_confidence: torch.Tensor
    raw: torch.Tensor

    def ycrcb(self, mode: str = "mean") -> torch.Tensor:
        if mode == "mean":
            chroma = self.chroma_mean
        elif mode == "safe":
            chroma = self.chroma_safe
        else:
            raise ValueError("V7 inference mode must be 'mean' or 'safe'")
        return torch.cat((self.luma, chroma), dim=1)


class V7PolarChromaRefiner(nn.Module):
    """V6 trunk with a five-map probabilistic polar output head."""

    model_version = "v7"
    model_family = "v7_polar_probabilistic"

    def __init__(self, config: V7Config | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if config is None:
            config = V7Config()
        elif not isinstance(config, V7Config):
            config = V7Config(**dict(config))
        config.validate()
        self.config = config
        self.head = nn.Conv2d(config.in_channels, config.width, 3, padding=1)
        self.body = nn.Sequential(
            *(ResBlock(config.width) for _ in range(config.depth))
        )
        self.tail = nn.Conv2d(config.width, 5, 3, padding=1)
        self.reset_polar_head()

    def reset_polar_head(self) -> None:
        """Initialize the V7 mean as identity/bilinear and uncertainty conservatively."""
        nn.init.zeros_(self.tail.weight)
        with torch.no_grad():
            self.tail.bias.zero_()
            self.tail.bias[1] = 1.0  # cos(delta phase)=1
            self.tail.bias[3] = inverse_softplus(self.config.initial_amplitude_scale)
            self.tail.bias[4] = inverse_softplus(
                self.config.initial_phase_concentration
            )

    def forward(self, inputs: torch.Tensor) -> V7Prediction:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError(f"Expected Bx3xHxW [Y,Cr,Cb], got {tuple(inputs.shape)}")
        if not torch.is_floating_point(inputs):
            raise TypeError("V7 input must be floating point")
        raw = self.tail(self.body(self.head(inputs)))
        delta_amplitude = raw[:, 0:1]
        phase_vector = raw[:, 1:3]
        squared_length = phase_vector[:, 0:1].square() + phase_vector[:, 1:2].square()
        zero_length = squared_length <= self.config.eps
        phase_cos_raw = torch.where(
            zero_length, torch.ones_like(phase_vector[:, 0:1]), phase_vector[:, 0:1]
        )
        phase_sin_raw = torch.where(
            zero_length, torch.zeros_like(phase_vector[:, 1:2]), phase_vector[:, 1:2]
        )
        vector_norm = torch.sqrt(
            phase_cos_raw.square() + phase_sin_raw.square() + self.config.eps
        )
        cos_delta = phase_cos_raw / vector_norm
        sin_delta = phase_sin_raw / vector_norm
        delta_phase = torch.atan2(sin_delta, cos_delta)

        amplitude_0, phase_0 = cartesian_chroma_to_polar(
            inputs[:, 1:3], self.config.neutral_chroma
        )
        phase_mean = wrap_angle(phase_0 + delta_phase)
        radial_limit = maximum_amplitude_for_phase(
            phase_mean, self.config.neutral_chroma, self.config.eps
        )
        amplitude_mean = torch.minimum(
            torch.clamp_min(amplitude_0 + delta_amplitude, 0.0), radial_limit
        )
        chroma_mean = polar_chroma_to_cartesian(
            amplitude_mean, phase_mean, self.config.neutral_chroma
        )

        # Special-function losses cast these float32 values explicitly; keeping
        # transforms here elementary makes forward safe under autocast.
        amplitude_scale = F.softplus(raw[:, 3:4].float()) + self.config.eps
        phase_kappa = (F.softplus(raw[:, 4:5].float()) + self.config.eps).clamp(
            self.config.kappa_min, self.config.kappa_max
        )
        amplitude_scale = amplitude_scale.to(raw.dtype)
        phase_kappa = phase_kappa.to(raw.dtype)

        amplitude_confidence = torch.exp(
            -amplitude_scale / self.config.safe_amplitude_scale_reference
        ).clamp(0.0, 1.0)
        phase_confidence = (
            phase_kappa / (phase_kappa + self.config.safe_phase_kappa_reference)
        ) * (
            amplitude_mean
            / (amplitude_mean + self.config.safe_phase_amplitude_reference)
        )
        phase_confidence = phase_confidence.clamp(0.0, 1.0)
        amplitude_safe = amplitude_0 + amplitude_confidence * (
            amplitude_mean - amplitude_0
        )
        phase_safe = wrap_angle(
            phase_0 + phase_confidence * circular_difference(phase_mean, phase_0)
        )
        safe_limit = maximum_amplitude_for_phase(
            phase_safe, self.config.neutral_chroma, self.config.eps
        )
        amplitude_safe = torch.minimum(amplitude_safe.clamp_min(0.0), safe_limit)
        chroma_safe = polar_chroma_to_cartesian(
            amplitude_safe, phase_safe, self.config.neutral_chroma
        )
        return V7Prediction(
            neutral_chroma=self.config.neutral_chroma,
            luma=inputs[:, 0:1],
            chroma_mean=chroma_mean,
            chroma_safe=chroma_safe,
            amplitude_baseline=amplitude_0,
            phase_baseline=phase_0,
            amplitude_mean=amplitude_mean,
            phase_mean=phase_mean,
            amplitude_scale=amplitude_scale,
            phase_kappa=phase_kappa,
            amplitude_confidence=amplitude_confidence,
            phase_confidence=phase_confidence,
            raw=raw,
        )


def parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def save_v7_checkpoint(
    path: str | Path,
    model: V7PolarChromaRefiner,
    epoch: int,
    seed: int,
    loss_config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    git_commit: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "format_version": 3,
        "model_version": "v7",
        "model_family": model.model_family,
        "architecture": asdict(model.config),
        "neutral_chroma": model.config.neutral_chroma,
        "loss": dict(loss_config),
        "uncertainty": {
            "amplitude_distribution": "laplace",
            "phase_distribution": "von_mises",
            "kappa_min": model.config.kappa_min,
            "kappa_max": model.config.kappa_max,
        },
        "training_seed": int(seed),
        "epoch": int(epoch),
        "git_commit": git_commit,
        "model": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(dict(extra))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_v7_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
    expected_config: V7Config | None = None,
) -> tuple[V7PolarChromaRefiner, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Invalid V7 checkpoint structure: {path}")
    if checkpoint.get("format_version") != 3:
        raise ValueError(
            f"Unsupported V7 checkpoint format: {checkpoint.get('format_version')}"
        )
    if (
        checkpoint.get("model_version") != "v7"
        or checkpoint.get("model_family") != V7PolarChromaRefiner.model_family
    ):
        raise ValueError(f"Not a compatible V7 checkpoint: {path}")
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, dict):
        raise TypeError("V7 checkpoint is missing architecture metadata")
    config = V7Config(**architecture)
    if float(checkpoint.get("neutral_chroma", -1.0)) != config.neutral_chroma:
        raise ValueError("V7 checkpoint neutral chroma metadata is inconsistent")
    if expected_config is not None and asdict(config) != asdict(expected_config):
        raise ValueError(
            "V7 checkpoint architecture does not match expected configuration"
        )
    for field in ("loss", "uncertainty", "training_seed", "epoch"):
        if field not in checkpoint:
            raise ValueError(f"V7 checkpoint is missing required metadata: {field}")
    model = V7PolarChromaRefiner(config).to(device)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise TypeError("V7 checkpoint is missing model state")
    model.load_state_dict(state, strict=True)
    return model, checkpoint
