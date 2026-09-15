#!/usr/bin/env python3
"""Train one or all independently named Prism experiments."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from chroma.prism_config import load_suite, experiment_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "research/configs/prism.json"
    )
    parser.add_argument(
        "--model", default="all", help="Experiment name, comma-separated names, or all"
    )
    parser.add_argument(
        "--list", action="store_true", help="List names; needs only standard Python"
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="auto resumes each run's last.pth; an explicit path must be that run's last.pth",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="End early without changing the planned scheduler horizon",
    )
    parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="Only for a manifest just hashed by the local job launcher; split checks remain enabled",
    )
    args = parser.parse_args()
    suite = load_suite(args.config)
    names = [row["name"] for row in suite["experiments"]]
    if args.list:
        print("\n".join(names))
        return
    chosen = names if args.model == "all" else args.model.split(",")
    if len(set(chosen)) != len(chosen) or any(name not in names for name in chosen):
        parser.error("Unknown or duplicate model; use --list")
    if args.resume and args.resume != "auto" and len(chosen) != 1:
        parser.error("An explicit resume checkpoint requires exactly one model")
    for argument, key in (
        ("manifest", "manifest"),
        ("dataset_root", "dataset_root"),
        ("output_root", "output_dir"),
    ):
        value = getattr(args, argument)
        if value is not None:
            suite[key] = str(value.resolve())
    for key in ("device", "epochs", "batch_size", "workers"):
        value = getattr(args, key)
        if value is not None:
            suite["training"][key] = value
    if args.skip_hash_verification:
        suite["training"]["verify_hashes"] = False
    # Import torch only after argument/list validation, so login-node listing is cheap.
    from chroma.prism_training import train_prism

    for name in chosen:
        result = train_prism(
            experiment_config(suite, name),
            PROJECT_ROOT,
            args.resume,
            args.stop_after_epoch,
        )
        print(json.dumps({"summary": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
