#!/usr/bin/env python3
"""Compare Prism checkpoints on identical held-out sources and degradations."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.limit < 0 or args.bootstrap_samples < 1:
        parser.error("limit must be nonnegative and bootstrap-samples positive")
    from chroma.prism_config import load_suite
    from chroma.prism_evaluation import evaluate_prism

    result = evaluate_prism(
        load_suite(args.config),
        args.weights,
        args.manifest,
        args.dataset_root,
        args.output_dir,
        args.device,
        args.split,
        args.limit,
        args.bootstrap_samples,
    )
    print(
        json.dumps(
            {
                "images": result["images"],
                "records": result["records"],
                "report": str(args.output_dir / "report.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
