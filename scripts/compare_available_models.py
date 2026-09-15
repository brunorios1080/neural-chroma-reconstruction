#!/usr/bin/env python3
"""Evaluate a frozen campaign of model families with resumable batched metrics."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.limit < 0:
        parser.error("Invalid batch size, workers, or limit")
    from chroma.model_comparison import compare
    compare(args.campaign, args.dataset_root, args.output, args.group, args.device,
            args.batch_size, args.workers, args.limit)
