#!/usr/bin/env python3
"""Run V7 procedural recoverability fixtures (not scientific evidence)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.v7_fixtures import write_hypothesis_fixture_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = write_hypothesis_fixture_report(args.weights, args.output, args.device)
    print(report)


if __name__ == "__main__":
    main()
