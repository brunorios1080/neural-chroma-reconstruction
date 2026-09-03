"""Synthetic behavioral fixtures for V7's recoverability hypothesis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .research_data import DegradationSpec, simulate_420
from .v7 import load_v7_checkpoint


def ambiguous_same_luma_pair(size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Targets with identical Y and identical box-420 observations but opposite chroma."""
    if size < 4 or size % 2:
        raise ValueError("Fixture size must be even and >=4")
    rows, columns = np.indices((size, size))
    checker = ((rows + columns) % 2).astype(np.float32)
    first = np.empty((size, size, 3), dtype=np.float32)
    first[..., 0] = 0.5
    first[..., 1] = 0.3 + 0.4 * checker
    first[..., 2] = 0.7 - 0.4 * checker
    second = first.copy()
    second[..., 1:3] = 1.0 - first[..., 1:3]
    return first, second


def neutral_fixture(size: int = 32) -> np.ndarray:
    image = np.full((size, size, 3), 0.5, dtype=np.float32)
    image[..., 0] = np.linspace(0.1, 0.9, size, dtype=np.float32)[None, :]
    return image


def hue_wrap_pair(size: int = 16, radius: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    targets = []
    for phase in (math.pi - 1e-4, -math.pi + 1e-4):
        image = np.full((size, size, 3), 0.5, dtype=np.float32)
        image[..., 1] = 0.5 + radius * math.cos(phase)
        image[..., 2] = 0.5 + radius * math.sin(phase)
        targets.append(image)
    return targets[0], targets[1]


def sharp_chroma_edge(size: int = 32) -> np.ndarray:
    image = np.full((size, size, 3), 0.5, dtype=np.float32)
    image[..., 0] = 0.5
    image[:, : size // 2, 1:3] = (0.25, 0.65)
    image[:, size // 2 :, 1:3] = (0.75, 0.35)
    return image


def evaluate_hypothesis_fixtures(
    checkpoint_path: str | Path, device: torch.device | str = "cpu"
) -> dict[str, Any]:
    """Measure behavior without turning procedural cases into scientific evidence."""
    model, checkpoint = load_v7_checkpoint(checkpoint_path, device)
    model.eval()
    spec = DegradationSpec("center_box", "center", "box", "bilinear")
    first, second = ambiguous_same_luma_pair()
    first_input, first_low = simulate_420(first, spec)
    second_input, second_low = simulate_420(second, spec)

    def predict(image: np.ndarray):
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.inference_mode():
            return model(tensor)

    first_prediction = predict(first_input)
    second_prediction = predict(second_input)
    neutral_prediction = predict(simulate_420(neutral_fixture(), spec)[0])
    edge_target = sharp_chroma_edge()
    edge_input = simulate_420(edge_target, spec)[0]
    edge_prediction = predict(edge_input)
    report = {
        "kind": "procedural_behavioral_fixture",
        "scientific_evidence": False,
        "checkpoint_epoch": checkpoint["epoch"],
        "ambiguous_observation_max_difference": float(
            np.max(np.abs(first_input - second_input))
        ),
        "ambiguous_lowres_max_difference": float(
            np.max(np.abs(first_low - second_low))
        ),
        "ambiguous_prediction_max_difference": float(
            torch.max(
                torch.abs(first_prediction.chroma_mean - second_prediction.chroma_mean)
            )
        ),
        "ambiguous_amplitude_scale_mean": float(
            first_prediction.amplitude_scale.mean()
        ),
        "neutral_phase_kappa_mean": float(neutral_prediction.phase_kappa.mean()),
        "neutral_reconstruction_mae": float(
            torch.mean(
                torch.abs(
                    neutral_prediction.chroma_mean
                    - torch.from_numpy(neutral_fixture()[..., 1:3])
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )
            )
        ),
        "sharp_edge_mean_chroma_mae": float(
            torch.mean(
                torch.abs(
                    edge_prediction.chroma_mean
                    - torch.from_numpy(edge_target[..., 1:3])
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )
            )
        ),
    }
    return report


def write_hypothesis_fixture_report(
    checkpoint_path: str | Path,
    output_path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    report = evaluate_hypothesis_fixtures(checkpoint_path, device)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
