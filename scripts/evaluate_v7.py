#!/usr/bin/env python3
"""Evaluate V7 mean/safe reconstruction and uncertainty on a manifest split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.v7_evaluation import run_v7_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, help="Override config checkpoint")
    parser.add_argument("--output-dir", type=Path, help="Override config output")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.weights:
        config["weights"] = str(args.weights)
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    report = run_v7_evaluation(config, PROJECT_ROOT)
    print(f"Wrote {report['records']} V7 image-condition-mode records")


if __name__ == "__main__":
    main()
