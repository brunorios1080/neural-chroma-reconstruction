#!/usr/bin/env python3
"""Build a deterministic manifest for a flat COCO image directory."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.research_data import ManifestRecord, sha256_file


def assign_splits(
    paths: list[Path], validation_fraction: float, test_fraction: float, seed: int
) -> dict[Path, str]:
    if validation_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("Split fractions cannot be negative")
    held_out_fraction = validation_fraction + test_fraction
    if held_out_fraction <= 0.0 or held_out_fraction >= 1.0:
        raise ValueError("Validation plus test fraction must be in (0, 1)")
    shuffled = sorted(paths)
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * (1.0 - held_out_fraction))
    test_count = int(len(shuffled) * test_fraction)
    validation_count = len(shuffled) - train_count - test_count
    if min(train_count, validation_count) < 1:
        raise ValueError("Dataset is too small for the requested train/validation split")
    assignments = {path: "train" for path in shuffled[:train_count]}
    validation_end = train_count + validation_count
    assignments.update(
        {path: "validation" for path in shuffled[train_count:validation_end]}
    )
    assignments.update({path: "test" for path in shuffled[validation_end:]})
    return assignments


def load_image_info(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(entry["file_name"]): entry for entry in payload["images"]}


def build_coco_manifest(
    images: Path,
    dataset_root: Path,
    output: Path,
    metadata: Path | None,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    max_images: int | None = None,
    min_size: int = 0,
) -> list[ManifestRecord]:
    images = images.resolve()
    dataset_root = dataset_root.resolve()
    if not images.is_dir():
        raise FileNotFoundError(f"COCO image directory does not exist: {images}")
    try:
        images.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError("Image directory must be inside dataset root") from error
    paths = sorted(
        path for path in images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg"}
    )
    if not paths:
        raise ValueError(f"No JPEG images found in {images}")
    image_info = load_image_info(metadata)
    if image_info:
        missing_metadata = sorted(
            path.name for path in paths if path.name not in image_info
        )
        if missing_metadata:
            raise ValueError(
                f"COCO metadata is missing {len(missing_metadata)} images; "
                f"first missing file: {missing_metadata[0]}"
            )
    if min_size < 0:
        raise ValueError("min_size cannot be negative")
    original_count = len(paths)
    if min_size:
        if image_info:
            paths = [
                path
                for path in paths
                if int(image_info[path.name]["width"]) >= min_size
                and int(image_info[path.name]["height"]) >= min_size
            ]
        else:
            eligible = []
            for path in paths:
                with Image.open(path) as image:
                    width, height = image.size
                if width >= min_size and height >= min_size:
                    eligible.append(path)
            paths = eligible
    excluded_too_small = original_count - len(paths)
    if max_images is not None:
        if max_images < 2:
            raise ValueError("max_images must be at least 2")
        paths = paths[:max_images]
    if len(paths) < 2:
        raise ValueError("Too few images remain after applying size and count filters")
    assignments = assign_splits(paths, validation_fraction, test_fraction, seed)
    records: list[ManifestRecord] = []
    for index, path in enumerate(paths, start=1):
        info = image_info.get(path.name)
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            image_format = image.format or "JPEG"
        if info is not None and (width, height) != (
            int(info["width"]),
            int(info["height"]),
        ):
            raise ValueError(f"COCO metadata dimensions do not match {path.name}")
        split = assignments[path]
        records.append(
            ManifestRecord(
                id=f"{split}/{path.stem}",
                relative_path=path.relative_to(dataset_root).as_posix(),
                split=split,
                sha256=sha256_file(path),
                width=width,
                height=height,
                mode=mode,
                image_format=image_format,
                source_group="coco_unlabeled2017",
            )
        )
        if index % 10000 == 0:
            print(f"Hashed {index}/{len(paths)} COCO images", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    split_counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "validation", "test")
        if any(record.split == split for record in records)
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "dataset": "COCO 2017 unlabeled",
                "records": len(records),
                "splits": split_counts,
                "seed": seed,
                "validation_fraction": validation_fraction,
                "test_fraction": test_fraction,
                "min_size": min_size,
                "excluded_too_small": excluded_too_small,
                "manifest_sha256": sha256_file(output),
                "lossless_only": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.03)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--max-images",
        type=int,
        help="Use only the first N sorted images for a small cluster smoke run",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=0,
        help="Exclude images whose width or height is smaller than this value",
    )
    args = parser.parse_args()
    records = build_coco_manifest(
        args.images,
        args.dataset_root,
        args.output,
        args.metadata,
        args.validation_fraction,
        args.test_fraction,
        args.seed,
        args.max_images,
        args.min_size,
    )
    counts = {
        split: sum(record.split == split for record in records)
        for split in sorted({record.split for record in records})
    }
    print(json.dumps({"records": len(records), "splits": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
