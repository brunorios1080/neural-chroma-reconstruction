#!/usr/bin/env python3
"""Run the manifest-driven expanded chroma reconstruction benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.research_benchmark import run_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_from_config(args.config, PROJECT_ROOT)
    print(
        f"Evaluated {report['evaluated_source_images']} source images and wrote "
        f"{report['per_image_records']} method-image-condition records."
    )


if __name__ == "__main__":
    main()
