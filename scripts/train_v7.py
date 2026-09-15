#!/usr/bin/env python3
"""Train V7 from a manifest-driven JSON configuration."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.v7_training import train_v7


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply cluster/job-specific overrides without editing a tracked config."""
    updated = copy.deepcopy(config)
    training = dict(updated.get("training", {}))
    for argument, key in (
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("workers", "workers"),
    ):
        value = getattr(args, argument)
        if value is not None:
            training[key] = value
    if args.device is not None:
        training["device"] = args.device
    if args.resume is not None:
        training["resume"] = str(args.resume)
    if args.no_amp:
        training["amp"] = False
    if args.skip_manifest_verification:
        training["verify_manifest"] = False
    updated["training"] = training
    for argument, key in (
        ("manifest", "manifest"),
        ("dataset_root", "dataset_root"),
        ("output_dir", "output_dir"),
    ):
        value = getattr(args, argument)
        if value is not None:
            updated[key] = str(value)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--skip-manifest-verification",
        action="store_true",
        help="Skip a redundant hash pass only when the manifest was just built locally",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train_v7(apply_overrides(config, args), PROJECT_ROOT)


if __name__ == "__main__":
    main()
