#!/usr/bin/env python3
"""Impute full-resolution image chroma from 4:2:0 samples with TabPFN V3."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.data import read_rgb, rgb_to_ycrcb, simulate_420
from chroma.metrics import image_metrics, ycrcb_to_rgb_float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--crop",
        type=int,
        default=64,
        help="Square center crop size; must be even (default: 64)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/tabpfn_v3_chroma")
    )
    parser.add_argument(
        "--model-path",
        default="v3_default",
        help="Hosted TabPFN model identifier (default: v3_default)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    if size < 8 or size % 2:
        raise ValueError("--crop must be an even integer of at least 8")
    height, width = image.shape[:2]
    if height < size or width < size:
        raise ValueError(f"image is {width}x{height}, smaller than {size}x{size}")
    top = (height - size) // 2
    left = (width - size) // 2
    return image[top : top + size, left : left + size]


def make_features(luma: np.ndarray) -> np.ndarray:
    """Build per-pixel spatial and luma features without using target chroma."""
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
        image, (width // 2, height // 2), interpolation=cv2.INTER_AREA
    )


def chroma_gradient_mae(reference: np.ndarray, candidate: np.ndarray) -> float:
    errors = []
    for channel in range(2):
        ref = reference[:, :, channel]
        pred = candidate[:, :, channel]
        for dx, dy in ((1, 0), (0, 1)):
            ref_gradient = cv2.Sobel(ref, cv2.CV_32F, dx, dy, ksize=3) / 8.0
            pred_gradient = cv2.Sobel(pred, cv2.CV_32F, dx, dy, ksize=3) / 8.0
            errors.append(np.abs(ref_gradient - pred_gradient))
    return float(np.mean(errors))


def evaluate(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    scores = image_metrics(reference, candidate)
    reference_chroma = reference[:, :, 1:3]
    candidate_chroma = candidate[:, :, 1:3]
    chroma_error = np.mean(np.abs(reference_chroma - candidate_chroma), axis=2)

    edge_strength = np.zeros(reference.shape[:2], dtype=np.float32)
    for channel in range(2):
        chroma = reference_chroma[:, :, channel]
        dx = cv2.Sobel(chroma, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        dy = cv2.Sobel(chroma, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        edge_strength += np.hypot(dx, dy)
    edge_mask = edge_strength >= np.percentile(edge_strength, 75.0)

    scores.update(
        {
            "chroma_mae": float(np.mean(chroma_error)),
            "chroma_edge_mae": float(np.mean(chroma_error[edge_mask])),
            "chroma_gradient_mae": chroma_gradient_mae(
                reference_chroma, candidate_chroma
            ),
        }
    )
    return scores


def save_rgb(path: Path, ycrcb: np.ndarray) -> None:
    rgb = np.rint(ycrcb_to_rgb_float(ycrcb) * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_comparison(
    path: Path,
    reference: np.ndarray,
    baseline: np.ndarray,
    prediction: np.ndarray,
) -> None:
    panels = []
    for label, image in (
        ("Original", reference),
        ("4:2:0 bilinear", baseline),
        ("TabPFN V3", prediction),
    ):
        rgb = np.rint(ycrcb_to_rgb_float(image) * 255.0).astype(np.uint8)
        enlarged = cv2.resize(rgb, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        panel = cv2.copyMakeBorder(
            enlarged, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)
        )
        cv2.putText(
            panel,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    comparison = np.concatenate(panels, axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
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
    target = rgb_to_ycrcb(center_crop(read_rgb(args.image), args.crop))
    baseline = simulate_420(target)
    features = make_features(target[:, :, 0])
    low_features = area_downsample(features)
    low_chroma = area_downsample(target[:, :, 1:3])
    train_x = low_features.reshape(-1, low_features.shape[2])
    test_x = features.reshape(-1, features.shape[2])

    predicted_channels = []
    for channel, name in enumerate(("Cr", "Cb")):
        print(f"Fitting {name} with {len(train_x)} observed chroma samples...")
        regressor = TabPFNRegressor(
            model_path=args.model_path,
            random_state=args.seed,
        )
        regressor.fit(train_x, low_chroma[:, :, channel].reshape(-1))
        prediction = np.asarray(regressor.predict(test_x), dtype=np.float32)
        predicted_channels.append(prediction.reshape(target.shape[:2]))

    tabpfn_chroma = np.stack(predicted_channels, axis=2)
    prediction = np.concatenate(
        (target[:, :, 0:1], np.clip(tabpfn_chroma, 0.0, 1.0)), axis=2
    )

    baseline_scores = evaluate(target, baseline)
    tabpfn_scores = evaluate(target, prediction)
    higher_is_better = {"rgb_psnr", "rgb_ssim", "chroma_psnr", "chroma_ssim"}
    improvement = {
        metric: (
            tabpfn_scores[metric] - baseline_scores[metric]
            if metric in higher_is_better
            else baseline_scores[metric] - tabpfn_scores[metric]
        )
        for metric in baseline_scores
    }
    report = {
        "image": str(args.image),
        "crop": args.crop,
        "model_path": args.model_path,
        "train_rows": len(train_x),
        "test_rows": len(test_x),
        "features": int(train_x.shape[1]),
        "baseline": baseline_scores,
        "tabpfn_v3": tabpfn_scores,
        "improvement": improvement,
        "improvement_sign": "positive values favor TabPFN V3",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(args.output_dir / "original.png", target)
    save_rgb(args.output_dir / "bilinear.png", baseline)
    save_rgb(args.output_dir / "tabpfn_v3.png", prediction)
    save_comparison(
        args.output_dir / "comparison.png", target, baseline, prediction
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Wrote experiment artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
