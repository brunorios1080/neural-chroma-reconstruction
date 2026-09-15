#!/usr/bin/env python3
"""Validate the packed COCO 2014 test set and create a test-only Prism manifest."""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile

from PIL import Image


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(archive, metadata_zip, output, exclude_manifests, min_size=256):
    if min_size < 1:
        raise ValueError("min_size must be positive")
    used_hashes, used_ids = set(), set()
    sources = []
    for path in exclude_manifests:
        path = Path(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["split"] in {"train", "validation"}:
                used_hashes.add(row["sha256"])
                used_ids.add(int(Path(row["relative_path"]).stem.split("_")[-1]))
        sources.append({"path": str(path), "sha256": file_hash(path)})
    with zipfile.ZipFile(metadata_zip) as info:
        metadata = json.loads(info.read("annotations/image_info_test2014.json"))
    images = {row["file_name"]: row for row in metadata["images"]}
    if len(images) != len(metadata["images"]):
        raise ValueError("Duplicate filenames in COCO metadata")
    rows, seen_hashes, excluded = [], set(), Counter()
    with zipfile.ZipFile(archive) as source:
        # Physical archive order avoids thousands of remote seeks on shared storage.
        names = [entry.filename for entry in sorted(source.infolist(), key=lambda entry: entry.header_offset)
                 if entry.filename.lower().endswith((".jpg", ".jpeg"))]
        expected = {f"test2014/{name}" for name in images}
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("Archive image entries do not match official metadata")
        for index, name in enumerate(names, 1):
            raw = source.read(name)  # Also checks the ZIP entry's CRC.
            digest = hashlib.sha256(raw).hexdigest()
            info = images[Path(name).name]
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
                mode, image_format = image.mode, image.format
                if (width, height) != (info["width"], info["height"]):
                    raise ValueError(f"Metadata dimension mismatch: {name}")
                image.verify()
            reason = None
            if digest in used_hashes or int(info["id"]) in used_ids:
                reason = "training_validation_overlap"
            elif min(width, height) < min_size:
                reason = "too_small"
            elif digest in seen_hashes:
                reason = "duplicate_content"
            if reason:
                excluded[reason] += 1
            else:
                seen_hashes.add(digest)
                rows.append({
                    "id": f"test/{Path(name).stem}", "relative_path": name,
                    "split": "test", "sha256": digest, "width": width,
                    "height": height, "mode": mode, "image_format": image_format,
                    "source_group": "coco_test2014",
                })
            if index % 10000 == 0:
                print(f"Verified {index}/{len(names)} test images", flush=True)
    if not rows:
        raise ValueError("No eligible independent test images remain")
    output = Path(output)
    rows.sort(key=lambda row: row["relative_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report = {
        "dataset": "COCO 2014 test", "baseline": "bilinear", "min_size": min_size,
        "archive": str(archive), "archive_sha256": file_hash(archive),
        "metadata_sha256": file_hash(metadata_zip), "archive_images": len(names),
        "records": len(rows), "splits": {"test": len(rows)},
        "excluded": {reason: excluded[reason] for reason in
                     ("training_validation_overlap", "too_small", "duplicate_content")},
        "exclusion_manifests": sources, "manifest_sha256": file_hash(output),
        "source_url": "https://s3.amazonaws.com/images.cocodataset.org/zips/test2014.zip",
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--metadata-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--min-size", type=int, default=256)
    args = parser.parse_args()
    prepare(args.archive, args.metadata_zip, args.output, args.exclude_manifests, args.min_size)
