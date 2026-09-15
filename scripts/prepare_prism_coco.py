#!/usr/bin/env python3
"""Hash, deduplicate, and split staged COCO images identically for all Prism jobs."""

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.build_coco_manifest import build_coco_manifest, assign_splits
from chroma.research_data import sha256_file


def prepare(
    images, root, output, metadata=None, seed=2026, min_size=256, max_images=None
):
    records = build_coco_manifest(
        images, root, output, metadata, 0.04, 0.02, seed, max_images, min_size
    )
    unique = {}
    for record in records:
        unique.setdefault(record.sha256, record)
    if len(unique) < 50:
        raise ValueError(
            "Prism COCO preparation requires >=50 distinct eligible images for a 4% validation/2% test split"
        )
    assignments = assign_splits(list(unique), 0.04, 0.02, seed)
    deduplicated = []
    for digest, record in unique.items():
        split = assignments[digest]
        deduplicated.append(
            replace(
                record, split=split, id=f"{split}/{Path(record.relative_path).stem}"
            )
        )
    output.write_text(
        "".join(json.dumps(asdict(r), sort_keys=True) + "\n" for r in deduplicated)
    )
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text())
    meta.update(
        records=len(deduplicated),
        duplicate_images_removed=len(records) - len(deduplicated),
        splits={
            split: sum(r.split == split for r in deduplicated)
            for split in ("train", "validation", "test")
        },
        manifest_sha256=sha256_file(output),
    )
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, sort_keys=True), flush=True)
    return deduplicated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()
    prepare(
        args.images,
        args.dataset_root,
        args.output,
        args.metadata,
        args.seed,
        args.min_size,
        args.max_images,
    )


if __name__ == "__main__":
    main()
