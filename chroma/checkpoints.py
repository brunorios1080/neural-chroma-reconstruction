"""Checkpoint save/load helpers, including legacy V5/V6 compatibility."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .models import normalize_version

try:
    from numpy._core.multiarray import scalar as numpy_scalar
except ImportError:  # NumPy 1.x
    from numpy.core.multiarray import scalar as numpy_scalar


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _torch_load(path: str | Path, map_location: torch.device | str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        # The legacy V5 checkpoint stores two NumPy scalar loss values. Allow
        # only the concrete NumPy globals required for those values, while
        # retaining the safer weights-only loader for user-supplied files.
        numpy_globals = [
            numpy_scalar,
            np.dtype,
            type(np.dtype(np.float64)),
            np.float64,
        ]
        with torch.serialization.safe_globals(numpy_globals):
            return torch.load(path, map_location=map_location, weights_only=True)


def model_state(checkpoint: dict[str, Any], version: str) -> dict[str, Any]:
    version = normalize_version(version)
    candidates = ("model", "G") if version == "v5" else ("model",)
    for key in candidates:
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return strip_module_prefix(state)
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return strip_module_prefix(checkpoint)
    raise ValueError(f"Checkpoint does not contain {version} model weights")


def load_model(
    model: nn.Module,
    path: str | Path,
    version: str,
    device: torch.device | str,
) -> dict[str, Any]:
    checkpoint = _torch_load(path, device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Invalid checkpoint structure: {path}")
    unwrap(model).load_state_dict(model_state(checkpoint, version), strict=True)
    return checkpoint


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
