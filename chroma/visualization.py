"""Correct YCrCb visualization helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .data import ycrcb_to_bgr_uint8


def tensor_to_ycrcb(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().permute(1, 2, 0).numpy()


def save_comparison(
    model_input: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    destination: str | Path,
) -> None:
    panels = [
        ycrcb_to_bgr_uint8(tensor_to_ycrcb(item))
        for item in (model_input, target, prediction)
    ]
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.hstack(panels)):
        raise OSError(f"Could not save comparison image: {path}")
