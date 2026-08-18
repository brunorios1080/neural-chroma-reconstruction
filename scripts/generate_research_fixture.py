#!/usr/bin/env python3
"""Generate deterministic lossless fixtures for local protocol validation only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def pattern(index: int, size: int) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    phase = index + 1
    red = (xx * (3 + phase) + yy * (phase % 3)) % 256
    green = (yy * (5 + phase) + 31 * phase) % 256
    blue = ((xx + yy) * (2 + phase) + 17 * phase) % 256
    image = np.stack((red, green, blue), axis=2).astype(np.uint8)
    boundary = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 < (size / (3 + index % 3)) ** 2
    color = np.asarray(
        ((47 * phase) % 256, (113 * phase) % 256, (197 * phase) % 256),
        dtype=np.uint8,
    )
    image[boundary] = color
    stripe = ((xx + phase * yy) % (9 + index % 5)) < 2
    image[stripe] = 255 - image[stripe]
    canvas = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (size // 8, size // 3, size - size // 7, size // 3 + 3),
        fill=(255, 20 + 11 * index % 220, 20),
    )
    draw.line(
        (0, size - 1 - index % 9, size - 1, index % 13),
        fill=(10, 240, 210),
        width=2,
    )
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("research/fixtures/lossless")
    )
    parser.add_argument("--size", type=int, default=96)
    args = parser.parse_args()
    splits = {"train": range(0, 6), "validation": range(6, 8), "test": range(8, 12)}
    for split, indices in splits.items():
        directory = args.output / split
        directory.mkdir(parents=True, exist_ok=True)
        for index in indices:
            Image.fromarray(pattern(index, args.size), mode="RGB").save(
                directory / f"fixture_{index:02d}.png"
            )
    print(
        f"Generated 12 deterministic lossless protocol fixtures under {args.output}. "
        "These images are not publication data."
    )


if __name__ == "__main__":
    main()
