#!/usr/bin/env python3
"""Build and verify a hash-addressed manifest for disjoint lossless splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.research_data import build_manifest, verify_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument(
        "--unlabeled",
        type=Path,
        help="Optional unlabeled-only split for V7 self-training manifests",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-lossy",
        action="store_true",
        help="Permit JPEG/WebP sources (not recommended for publication runs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.unlabeled is None and (args.train is None or args.test is None):
        raise ValueError("Supervised manifests require both --train and --test")
    splits = {}
    if args.train is not None:
        splits["train"] = args.train
    if args.test is not None:
        splits["test"] = args.test
    if args.validation is not None:
        splits["validation"] = args.validation
    if args.unlabeled is not None:
        splits["unlabeled"] = args.unlabeled
    records = build_manifest(
        splits,
        args.output,
        dataset_root=args.dataset_root,
        allow_lossy=args.allow_lossy,
    )
    verification = verify_manifest(records, args.dataset_root)
    print(json.dumps(verification, indent=2, sort_keys=True))
    if not verification["ok"]:
        raise SystemExit(1)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
