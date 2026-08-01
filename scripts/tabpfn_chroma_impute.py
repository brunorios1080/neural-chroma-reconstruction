#!/usr/bin/env python3
"""Impute full-resolution image chroma from 4:2:0 samples with TabPFN V3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.data import read_rgb, rgb_to_ycrcb, simulate_420
from chroma.metrics import reconstruction_metrics, ycrcb_to_rgb_float
from chroma.tabpfn import reconstruct


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
    target = rgb_to_ycrcb(center_crop(read_rgb(args.image), args.crop))
    baseline = simulate_420(target)
    prediction, metadata = reconstruct(target, args.model_path, args.seed)

    baseline_scores = reconstruction_metrics(target, baseline)
    tabpfn_scores = reconstruction_metrics(target, prediction)
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
        **metadata,
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
