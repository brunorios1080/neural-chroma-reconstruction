"""Polar helpers for normalized Cr/Cb tensors in repository channel order."""

from __future__ import annotations

import math

import torch

PUBLICATION_NEUTRAL_CHROMA = 0.5
OPENCV_UINT8_NEUTRAL_CHROMA = 128.0 / 255.0


def _validate_chroma(chroma: torch.Tensor) -> None:
    if chroma.ndim < 3 or chroma.shape[-3] != 2:
        raise ValueError(
            "Expected chroma with channel dimension ...x2xHxW in [Cr, Cb] order; "
            f"got {tuple(chroma.shape)}"
        )


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap radians to the half-open interval [-pi, pi)."""
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def circular_difference(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return the signed shortest difference ``first - second`` in radians."""
    return wrap_angle(first - second)


def cartesian_chroma_to_polar(
    chroma: torch.Tensor,
    neutral_chroma: float,
    eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert `[Cr, Cb]` to magnitude and phase without clipping.

    ``eps`` is optional and defaults to zero so the conversion has an exact
    round trip, including neutral gray. Phase at exactly zero magnitude follows
    PyTorch's well-defined ``atan2(0, 0) == 0`` convention and must be masked in
    any physical hue loss.
    """
    _validate_chroma(chroma)
    if not 0.0 <= neutral_chroma <= 1.0:
        raise ValueError("neutral_chroma must be in [0, 1]")
    if eps < 0.0:
        raise ValueError("eps cannot be negative")
    u = chroma[..., 0:1, :, :] - neutral_chroma
    v = chroma[..., 1:2, :, :] - neutral_chroma
    amplitude = torch.sqrt(torch.clamp(u.square() + v.square() + eps, min=0.0))
    phase = torch.atan2(v, u)
    return amplitude, phase


def polar_chroma_to_cartesian(
    amplitude: torch.Tensor,
    phase: torch.Tensor,
    neutral_chroma: float,
    clip: bool = False,
) -> torch.Tensor:
    """Convert magnitude/phase to `[Cr, Cb]`; clipping is opt-in."""
    if amplitude.shape != phase.shape:
        raise ValueError(
            f"Amplitude and phase shapes differ: {amplitude.shape} vs {phase.shape}"
        )
    if not 0.0 <= neutral_chroma <= 1.0:
        raise ValueError("neutral_chroma must be in [0, 1]")
    chroma = torch.cat(
        (
            neutral_chroma + amplitude * torch.cos(phase),
            neutral_chroma + amplitude * torch.sin(phase),
        ),
        dim=-3,
    )
    return chroma.clamp(0.0, 1.0) if clip else chroma


def maximum_amplitude_for_phase(
    phase: torch.Tensor, neutral_chroma: float, eps: float = 1e-8
) -> torch.Tensor:
    """Distance from the neutral point to the `[0,1]^2` boundary along phase."""
    if not 0.0 <= neutral_chroma <= 1.0:
        raise ValueError("neutral_chroma must be in [0, 1]")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    cosine = torch.cos(phase)
    sine = torch.sin(phase)
    # torch.where evaluates both branches. Dividing by the raw trigonometric
    # values therefore creates infinities at cardinal phases even when that
    # branch is not selected, and autograd can turn 0 * inf into NaN. Use safe
    # denominators and a finite upper bound for inactive axes instead.
    far_limit = torch.full_like(phase, math.sqrt(2.0))
    x_limit = torch.where(
        cosine > eps,
        (1.0 - neutral_chroma) / cosine.clamp_min(eps),
        torch.where(
            cosine < -eps,
            -neutral_chroma / cosine.clamp_max(-eps),
            far_limit,
        ),
    )
    y_limit = torch.where(
        sine > eps,
        (1.0 - neutral_chroma) / sine.clamp_min(eps),
        torch.where(
            sine < -eps,
            -neutral_chroma / sine.clamp_max(-eps),
            far_limit,
        ),
    )
    return torch.minimum(x_limit, y_limit).clamp_min(0.0)
