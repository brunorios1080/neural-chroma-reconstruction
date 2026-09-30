#!/usr/bin/env python3
"""Run fresh V5 generator training against a completed Prism MSE data protocol."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    from chroma.v5_prism_mse import train

    result = train(args.reference_run, args.dataset_root, args.output_dir,
                   args.workers, args.resume, args.stop_after_epoch, args.device)
    print(json.dumps({"summary": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
