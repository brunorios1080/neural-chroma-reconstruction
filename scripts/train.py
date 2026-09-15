#!/usr/bin/env python3
"""Train either the V5 GAN or V6 residual chroma model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.training import TrainingConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("v5", "v6"), required=True)
    parser.add_argument(
        "--src", type=Path, required=True, help="Directory containing source images"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--batch-size", "--batch", dest="batch_size", type=int, default=16
    )
    parser.add_argument("--crop", type=int, default=256, help="Square training crop")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or a device such as cuda:1"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir", "--out", dest="output_dir", type=Path)
    parser.add_argument("--samples-dir", "--samples", dest="samples_dir", type=Path)
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable CUDA mixed precision"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or PROJECT_ROOT / "runs" / args.model / "checkpoints"
    samples_dir = args.samples_dir or PROJECT_ROOT / "runs" / args.model / "samples"
    run_training(
        TrainingConfig(
            model_version=args.model,
            source=args.src,
            output_dir=output_dir,
            samples_dir=samples_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            crop_size=args.crop,
            workers=args.workers,
            val_fraction=args.val_fraction,
            seed=args.seed,
            learning_rate=args.learning_rate,
            device=args.device,
            resume=args.resume,
            amp=not args.no_amp,
            max_images=args.max_images,
            run_name=args.run_name,
        )
    )


if __name__ == "__main__":
    main()
