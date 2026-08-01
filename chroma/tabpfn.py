"""TabPFN-backed image-adaptive chroma reconstruction."""

from __future__ import annotations

import os
from collections.abc import Callable

import cv2
import numpy as np


def make_features(luma: np.ndarray) -> np.ndarray:
    """Build per-pixel spatial and luma features without target chroma."""
    height, width = luma.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    x = (xx + 0.5) / width * 2.0 - 1.0
    y = (yy + 0.5) / height * 2.0 - 1.0

    blur_3 = cv2.GaussianBlur(luma, (3, 3), 0)
    blur_7 = cv2.GaussianBlur(luma, (7, 7), 0)
    dx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    dy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gradient = np.hypot(dx, dy)

    return np.stack(
        (
            x,
            y,
            x * y,
            x * x,
            y * y,
            np.sin(np.pi * x),
            np.cos(np.pi * x),
            np.sin(np.pi * y),
            np.cos(np.pi * y),
            luma,
            blur_3,
            blur_7,
            luma - blur_3,
            dx,
            dy,
            gradient,
        ),
        axis=2,
    ).astype(np.float32)


def area_downsample(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(
        image, ((width + 1) // 2, (height + 1) // 2), interpolation=cv2.INTER_AREA
    )


def reconstruct(
    target_ycrcb: np.ndarray,
    model_path: str = "v3_default",
    seed: int = 0,
    status: Callable[[str], None] | None = print,
) -> tuple[np.ndarray, dict[str, int | str]]:
    """Reconstruct Cr/Cb with separate hosted regressors and preserve luma."""
    token = os.environ.get("TABPFN_TOKEN")
    if not token:
        raise RuntimeError("TABPFN_TOKEN is not set in the active environment")

    try:
        import tabpfn_client
        from tabpfn_client import TabPFNRegressor
    except ImportError as error:
        raise RuntimeError(
            "Install the hosted client with: pip install --upgrade tabpfn-client"
        ) from error

    tabpfn_client.set_access_token(token)
    features = make_features(target_ycrcb[:, :, 0])
    low_features = area_downsample(features)
    low_chroma = area_downsample(target_ycrcb[:, :, 1:3])
    train_x = low_features.reshape(-1, low_features.shape[2])
    test_x = features.reshape(-1, features.shape[2])

    predicted_channels = []
    for channel, name in enumerate(("Cr", "Cb")):
        if status is not None:
            status(f"Fitting {name} with {len(train_x)} observed chroma samples...")
        regressor = TabPFNRegressor(model_path=model_path, random_state=seed)
        regressor.fit(train_x, low_chroma[:, :, channel].reshape(-1))
        prediction = np.asarray(regressor.predict(test_x), dtype=np.float32)
        predicted_channels.append(prediction.reshape(target_ycrcb.shape[:2]))

    tabpfn_chroma = np.stack(predicted_channels, axis=2)
    prediction = np.concatenate(
        (target_ycrcb[:, :, 0:1], np.clip(tabpfn_chroma, 0.0, 1.0)), axis=2
    )
    metadata: dict[str, int | str] = {
        "model_path": model_path,
        "train_rows": len(train_x),
        "test_rows": len(test_x),
        "features": int(train_x.shape[1]),
    }
    return prediction, metadata
