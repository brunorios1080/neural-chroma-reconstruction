#!/usr/bin/env python3
"""Validate and copy supported images into a clean dataset directory."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.data import list_image_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--min-size", type=int, default=256, help="Minimum image width and height"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    destination = args.output.resolve()
    if source == destination:
        raise ValueError("Input and output directories must be different")
    files = [
        path for path in list_image_files(source) if destination not in path.parents
    ]
    copied = skipped = 0
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < args.min_size:
            skipped += 1
            continue
        output_path = destination / path.relative_to(source)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output_path)
        copied += 1
    print(f"Prepared {copied} images in {destination}; skipped {skipped}")
    if copied == 0:
        raise RuntimeError("No usable images were copied")


if __name__ == "__main__":
    main()
