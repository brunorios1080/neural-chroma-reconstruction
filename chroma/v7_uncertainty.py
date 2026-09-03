"""Prediction intervals and recoverability diagnostics for V7 evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

SUPPORTED_COVERAGES = (0.50, 0.80, 0.90, 0.95)


def laplace_interval(
    mean: torch.Tensor,
    scale: torch.Tensor,
    coverage: float,
    lower_bound: float = 0.0,
    upper_bound: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact central interval for a non-truncated Laplace distribution.

    Bounds are clipped to the physical amplitude domain after constructing the
    distributional interval; evaluation reports the resulting empirical coverage.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be in (0,1)")
    half_width = -scale * math.log(1.0 - coverage)
    lower = (mean - half_width).clamp_min(lower_bound)
    upper = mean + half_width
    if upper_bound is not None:
        if not torch.is_tensor(upper_bound):
            upper_bound = torch.as_tensor(
                upper_bound, dtype=upper.dtype, device=upper.device
            )
        upper = torch.minimum(upper, upper_bound)
    return lower, upper


def circular_variance(kappa: torch.Tensor) -> torch.Tensor:
    """Von Mises circular variance `1 - I1(kappa)/I0(kappa)` stably."""
    values = kappa.float().clamp_min(0.0)
    if not hasattr(torch.special, "i0e") or not hasattr(torch.special, "i1"):
        raise RuntimeError("V7 circular variance requires torch.special.i0e/i1")
    # i1e is present in supported PyTorch; retain a stable fallback for older 2.x.
    if hasattr(torch.special, "i1e"):
        ratio = torch.special.i1e(values) / torch.special.i0e(values).clamp_min(1e-30)
    else:
        safe = values.clamp_max(40.0)
        ratio = torch.special.i1(safe) / torch.special.i0(safe).clamp_min(1e-30)
    return (1.0 - ratio).clamp(0.0, 1.0).to(kappa.dtype)


@dataclass(frozen=True)
class VonMisesIntervalLookup:
    """Numerical symmetric central angular intervals; evaluation-only approximation."""

    kappa_grid: np.ndarray
    half_width_grid: np.ndarray
    coverage: float

    @classmethod
    def build(
        cls,
        coverage: float,
        kappa_max: float = 100.0,
        kappa_points: int = 512,
        angle_points: int = 8193,
    ) -> VonMisesIntervalLookup:
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be in (0,1)")
        if kappa_max <= 0.0 or kappa_points < 2 or angle_points < 101:
            raise ValueError("Invalid von Mises lookup resolution")
        kappa = np.concatenate(([0.0], np.geomspace(1e-4, kappa_max, kappa_points - 1)))
        angles = np.linspace(0.0, math.pi, angle_points, dtype=np.float64)
        half_widths = np.empty_like(kappa)
        for index, value in enumerate(kappa):
            # Symmetry gives P(|theta|<=h); subtracting kappa avoids overflow.
            density = np.exp(value * (np.cos(angles) - 1.0))
            increments = 0.5 * (density[1:] + density[:-1]) * np.diff(angles)
            cumulative = np.concatenate(([0.0], np.cumsum(increments)))
            cumulative /= cumulative[-1]
            half_widths[index] = np.interp(coverage, cumulative, angles)
        return cls(kappa, half_widths, coverage)

    def half_width(self, kappa: np.ndarray | torch.Tensor):
        values = (
            kappa.detach().float().cpu().numpy()
            if torch.is_tensor(kappa)
            else np.asarray(kappa, dtype=np.float64)
        )
        result = np.interp(
            np.clip(values, self.kappa_grid[0], self.kappa_grid[-1]),
            self.kappa_grid,
            self.half_width_grid,
        )
        if torch.is_tensor(kappa):
            return torch.as_tensor(result, dtype=kappa.dtype, device=kappa.device)
        return result


def spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Spearman correlation without a SciPy dependency, including average ties."""
    x = np.asarray(first, dtype=np.float64).ravel()
    y = np.asarray(second, dtype=np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        return float("nan")

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranked = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            stop = start + 1
            while stop < values.size and values[order[stop]] == values[order[start]]:
                stop += 1
            ranked[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
            start = stop
        return ranked

    rx, ry = ranks(x), ranks(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def risk_coverage_curve(
    error: np.ndarray,
    confidence: np.ndarray,
    coverages: Iterable[float] = (0.10, 0.25, 0.50, 0.75, 1.0),
) -> list[dict[str, float]]:
    """Return mean error when retaining the most confident pixels."""
    errors = np.asarray(error, dtype=np.float64).ravel()
    scores = np.asarray(confidence, dtype=np.float64).ravel()
    finite = np.isfinite(errors) & np.isfinite(scores)
    errors, scores = errors[finite], scores[finite]
    if errors.size == 0:
        return []
    order = np.argsort(-scores, kind="mergesort")
    result = []
    for coverage in coverages:
        if not 0.0 < coverage <= 1.0:
            raise ValueError("Risk coverage values must be in (0,1]")
        count = max(1, math.ceil(coverage * errors.size))
        retained = order[:count]
        result.append(
            {
                "coverage": float(count / errors.size),
                "requested_coverage": float(coverage),
                "mean_error": float(np.mean(errors[retained])),
                "confidence_threshold": float(np.min(scores[retained])),
                "pixels": int(count),
            }
        )
    return result


def empirical_interval_coverage(
    target: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    target = np.asarray(target)
    return float(np.mean((target >= np.asarray(lower)) & (target <= np.asarray(upper))))
