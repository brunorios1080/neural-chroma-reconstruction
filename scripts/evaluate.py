#!/usr/bin/env python3
"""Compare a trained model with bilinear 4:2:0 upsampling on an image set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.checkpoints import load_model
from chroma.data import YUVChromaDataset, list_image_files
from chroma.inference import predict
from chroma.metrics import image_metrics
from chroma.models import build_model
from chroma.training import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("v5", "v6"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument(
        "--crop", type=int, default=256, help="Center crop; use 0 for full images"
    )
    parser.add_argument(
        "--limit", type=int, help="Evaluate at most this many sorted images"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--psnr-cap", type=float, default=80.0)
    parser.add_argument("--json", type=Path, help="Optional JSON report destination")
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    files = list_image_files(args.src)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        files = files[: args.limit]
    crop_size = args.crop or None
    dataset = YUVChromaDataset(files, crop_size, random_crop=False)
    device = resolve_device(args.device)
    model = build_model(args.model).to(device)
    load_model(model, args.weights, args.model, device)
    model.eval()

    collected: dict[str, dict[str, list[float]]] = {
        "bilinear": defaultdict(list),
        "model": defaultdict(list),
    }
    skipped = []
    for index in tqdm(range(len(dataset)), desc=f"Evaluating {args.model.upper()}"):
        sample = dataset[index]
        if sample[0] is None:
            skipped.append({"path": sample[1], "reason": sample[2]})
            continue
        model_input, target, _ = sample
        prediction = (
            predict(model, model_input.unsqueeze(0).to(device), args.model)[0]
            .float()
            .cpu()
        )
        target_np = target.permute(1, 2, 0).numpy()
        baseline_np = model_input.permute(1, 2, 0).numpy()
        prediction_np = prediction.permute(1, 2, 0).numpy()
        for method, candidate in (("bilinear", baseline_np), ("model", prediction_np)):
            for metric, value in image_metrics(
                target_np, candidate, args.psnr_cap
            ).items():
                collected[method][metric].append(value)
    if not collected["model"]:
        raise RuntimeError("No images could be evaluated")

    report = {
        "model_version": args.model,
        "weights": str(args.weights),
        "images": len(next(iter(collected["model"].values()))),
        "skipped": skipped,
        "metrics": {
            method: {metric: _summary(values) for metric, values in metrics.items()}
            for method, metrics in collected.items()
        },
    }
    print(f"Evaluated {report['images']} images; skipped {len(skipped)}")
    print(f"{'METRIC':<16} {'BILINEAR':>12} {'MODEL':>12} {'DELTA':>12}")
    for metric in ("rgb_psnr", "rgb_ssim", "chroma_psnr", "chroma_ssim"):
        baseline = report["metrics"]["bilinear"][metric]["mean"]
        model_score = report["metrics"]["model"][metric]["mean"]
        print(
            f"{metric:<16} {baseline:>12.4f} {model_score:>12.4f} {model_score - baseline:>+12.4f}"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved report to {args.json}")


if __name__ == "__main__":
    main()
