#!/usr/bin/env python3
"""Run the controlled V6/V7 output-parameterization experiment matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.v7_training import run_v7_ablation_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.config.read_text(encoding="utf-8"))
    run_v7_ablation_matrix(matrix, PROJECT_ROOT)


if __name__ == "__main__":
    main()
