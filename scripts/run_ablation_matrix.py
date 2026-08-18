#!/usr/bin/env python3
"""Train the preregistered luma/residual/depth/width/loss ablation matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.research_benchmark import resolve_device
from chroma.research_data import load_manifest
from chroma.research_training import run_ablation_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    args = parse_args()
    matrix = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = resolve(matrix["manifest"], PROJECT_ROOT)
    dataset_root = resolve(matrix["dataset_root"], PROJECT_ROOT)
    output_dir = resolve(matrix["output_dir"], PROJECT_ROOT)
    run_ablation_matrix(
        matrix,
        load_manifest(manifest_path),
        dataset_root,
        output_dir,
        resolve_device(args.device),
    )


if __name__ == "__main__":
    main()
