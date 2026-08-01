#!/usr/bin/env python3
"""Paired COCO benchmark for interpolation, V5, V6, and TabPFN V3."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.checkpoints import load_model
from chroma.data import (
    list_image_files,
    read_rgb,
    rgb_to_ycrcb,
    simulate_420,
    to_tensor,
)
from chroma.inference import predict
from chroma.metrics import reconstruction_metrics, ycrcb_to_rgb_float
from chroma.models import build_model
from chroma.tabpfn import reconstruct as reconstruct_tabpfn
from chroma.training import resolve_device

METHODS = ("bilinear", "bicubic", "v5", "v6", "tabpfn_v3")
HIGHER_IS_BETTER = {"rgb_psnr", "rgb_ssim", "chroma_psnr", "chroma_ssim"}
PRIMARY_METRICS = (
    "rgb_psnr",
    "rgb_ssim",
    "chroma_psnr",
    "chroma_ssim",
    "chroma_mae",
    "chroma_edge_mae",
    "chroma_gradient_mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("data/raw/coco/val2017"))
    parser.add_argument("--images", type=int, default=20)
    parser.add_argument("--crop", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--v5-weights", type=Path, default=Path("models/version5/epoch_010.pth")
    )
    parser.add_argument(
        "--v6-weights",
        type=Path,
        default=Path("models/version6/res_epoch_030.pth"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-path", default="v3_default")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/coco_tabpfn_benchmark")
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute completed image records and consume API quota again",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.images < 2:
        raise ValueError("--images must be at least 2")
    if args.crop < 8 or args.crop % 2:
        raise ValueError("--crop must be an even integer of at least 8")
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")


def select_crops(
    source: Path, count: int, crop_size: int, seed: int
) -> list[tuple[Path, int, int, np.ndarray]]:
    files = list_image_files(source)
    rng = random.Random(seed)
    rng.shuffle(files)
    selected = []
    for path in files:
        try:
            image = read_rgb(path)
        except (OSError, ValueError, cv2.error):
            continue
        height, width = image.shape[:2]
        if height < crop_size or width < crop_size:
            continue
        top = rng.randint(0, height - crop_size)
        left = rng.randint(0, width - crop_size)
        crop = image[top : top + crop_size, left : left + crop_size]
        selected.append((path, top, left, crop))
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Only found {len(selected)} images large enough for the crop")


def neural_prediction(
    model: torch.nn.Module,
    model_input: np.ndarray,
    version: str,
    device: torch.device,
) -> np.ndarray:
    inputs = to_tensor(model_input).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = predict(model, inputs, version)[0].float().cpu()
    return output.permute(1, 2, 0).numpy()


def save_ycrcb(path: Path, image: np.ndarray) -> None:
    rgb = np.rint(ycrcb_to_rgb_float(image) * 255.0).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write image: {path}")


def directional_delta(metric: str, candidate: float, baseline: float) -> float:
    if metric in HIGHER_IS_BETTER:
        return candidate - baseline
    return baseline - candidate


def bootstrap_interval(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> list[float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return [float(low), float(high)]


def compare_methods(
    records: list[dict],
    candidate: str,
    reference: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict]:
    """Return paired directional improvements; positive favors the candidate."""
    metric_names = tuple(records[0]["metrics"][reference])
    rng = np.random.default_rng(seed)
    comparison = {}
    for metric in metric_names:
        deltas = np.asarray(
            [
                directional_delta(
                    metric,
                    record["metrics"][candidate][metric],
                    record["metrics"][reference][metric],
                )
                for record in records
            ],
            dtype=np.float64,
        )
        comparison[metric] = {
            "mean_improvement": float(np.mean(deltas)),
            "median_improvement": float(np.median(deltas)),
            "bootstrap_95_ci": bootstrap_interval(deltas, bootstrap_samples, rng),
            "win_rate": float(np.mean(deltas > 0.0)),
            "tie_rate": float(np.mean(deltas == 0.0)),
        }
    return comparison


def summarize(
    records: list[dict], bootstrap_samples: int, seed: int
) -> tuple[dict, dict]:
    metric_names = tuple(records[0]["metrics"]["bilinear"])
    aggregate: dict[str, dict] = {}
    for method in METHODS:
        aggregate[method] = {}
        for metric in metric_names:
            values = np.asarray(
                [record["metrics"][method][metric] for record in records],
                dtype=np.float64,
            )
            aggregate[method][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    paired: dict[str, dict] = {}
    for index, method in enumerate(METHODS[1:], start=1):
        paired[method] = compare_methods(
            records,
            candidate=method,
            reference="bilinear",
            bootstrap_samples=bootstrap_samples,
            seed=seed + index,
        )
    return aggregate, paired


def save_contact_sheet(output_dir: Path, records: list[dict]) -> None:
    scale = 2
    label_height = 24
    rows = []
    for record in records[:8]:
        record_dir = output_dir / "images" / record["record_id"]
        cells = []
        for method in ("original", *METHODS):
            bgr = cv2.imread(str(record_dir / f"{method}.png"), cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError(f"Could not read contact-sheet image for {method}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            enlarged = cv2.resize(
                rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
            )
            cell = cv2.copyMakeBorder(
                enlarged,
                label_height,
                0,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(25, 25, 25),
            )
            label = method.replace("_", " ").title()
            if method == "original":
                label = f"Original {record['record_id']}"
            cv2.putText(
                cell,
                label,
                (5, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cells.append(cell)
        rows.append(np.concatenate(cells, axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(
        str(output_dir / "contact_sheet.png"),
        cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    models = {}
    for version, weights in (("v5", args.v5_weights), ("v6", args.v6_weights)):
        model = build_model(version).to(device)
        load_model(model, weights, version, device)
        model.eval()
        models[version] = model

    selections = select_crops(args.src, args.images, args.crop, args.seed)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (path, top, left, rgb_crop) in enumerate(selections, start=1):
        record_id = f"{index:03d}_{path.stem}"
        record_dir = images_dir / record_id
        record_path = record_dir / "record.json"
        if record_path.is_file() and not args.overwrite:
            expected_crop = {"top": top, "left": left, "size": args.crop}
            cached = json.loads(record_path.read_text(encoding="utf-8"))
            cache_matches = (
                cached.get("image") == str(path)
                and cached.get("crop") == expected_crop
                and cached.get("tabpfn", {}).get("model_path") == args.model_path
            )
            if not cache_matches:
                raise RuntimeError(
                    f"Cached record {record_path} does not match this run; "
                    "use a new --output-dir or pass --overwrite"
                )
            print(f"[{index}/{args.images}] Reusing {record_id}")
            records.append(cached)
            continue

        print(f"[{index}/{args.images}] Evaluating {record_id}")
        target = rgb_to_ycrcb(rgb_crop)
        bilinear = simulate_420(target, cv2.INTER_LINEAR)
        candidates = {
            "bilinear": bilinear,
            "bicubic": simulate_420(target, cv2.INTER_CUBIC),
            "v5": neural_prediction(models["v5"], bilinear, "v5", device),
            "v6": neural_prediction(models["v6"], bilinear, "v6", device),
        }
        candidates["tabpfn_v3"], tabpfn_metadata = reconstruct_tabpfn(
            target,
            model_path=args.model_path,
            seed=args.seed,
            status=lambda message: print(f"  {message}"),
        )
        record = {
            "record_id": record_id,
            "image": str(path),
            "crop": {"top": top, "left": left, "size": args.crop},
            "tabpfn": tabpfn_metadata,
            "metrics": {
                method: reconstruction_metrics(target, candidate)
                for method, candidate in candidates.items()
            },
        }
        record_dir.mkdir(parents=True, exist_ok=True)
        save_ycrcb(record_dir / "original.png", target)
        for method, candidate in candidates.items():
            save_ycrcb(record_dir / f"{method}.png", candidate)
        record_path.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        records.append(record)

    aggregate, paired = summarize(records, args.bootstrap_samples, args.seed)
    tabpfn_comparisons = {
        method: compare_methods(
            records,
            candidate="tabpfn_v3",
            reference=method,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 100 + index,
        )
        for index, method in enumerate(METHODS[:-1])
    }
    report = {
        "benchmark": "paired random COCO chroma reconstruction",
        "config": {
            "source": str(args.src),
            "images": args.images,
            "crop": args.crop,
            "seed": args.seed,
            "device": str(device),
            "v5_weights": str(args.v5_weights),
            "v6_weights": str(args.v6_weights),
            "tabpfn_model_path": args.model_path,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "metric_direction": {
            metric: "higher" if metric in HIGHER_IS_BETTER else "lower"
            for metric in PRIMARY_METRICS
        },
        "aggregate": aggregate,
        "paired_vs_bilinear": paired,
        "tabpfn_v3_vs_method": tabpfn_comparisons,
        "records": records,
    }
    report_path = args.output_dir / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save_contact_sheet(args.output_dir, records)

    print(f"\n{'METHOD':<12} {'CHROMA PSNR':>13} {'DELTA':>10} {'WIN RATE':>10}")
    for method in METHODS:
        mean = aggregate[method]["chroma_psnr"]["mean"]
        if method == "bilinear":
            print(f"{method:<12} {mean:>13.4f} {'--':>10} {'--':>10}")
            continue
        comparison = paired[method]["chroma_psnr"]
        print(
            f"{method:<12} {mean:>13.4f} "
            f"{comparison['mean_improvement']:>+10.4f} "
            f"{comparison['win_rate']:>9.1%}"
        )
    print(f"Saved benchmark to {report_path}")


if __name__ == "__main__":
    main()
