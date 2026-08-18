"""Training and inference utilities for neural chroma reconstruction.

Model exports are loaded lazily so data-only tools such as manifest generation
do not initialize PyTorch (or its OpenMP runtime) unnecessarily.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ChromaRefiner", "Discriminator", "UNetGenerator", "build_model"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import models

        return getattr(models, name)
    raise AttributeError(name)
