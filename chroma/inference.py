"""Model inference helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .models import normalize_version


def predict(model: nn.Module, inputs: torch.Tensor, version: str) -> torch.Tensor:
    """Predict while padding V5 inputs to dimensions compatible with its U-Net."""
    version = normalize_version(version)
    if version != "v5":
        return model(inputs)
    height, width = inputs.shape[-2:]
    pad_height = (-height) % 8
    pad_width = (-width) % 8
    if pad_height or pad_width:
        inputs = F.pad(inputs, (0, pad_width, 0, pad_height), mode="replicate")
    return model(inputs)[..., :height, :width]
