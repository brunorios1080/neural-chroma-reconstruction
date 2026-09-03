#!/usr/bin/env python3
"""Run the optional V7 EMA teacher/student pseudo-label stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.v7_self_training import run_v7_self_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Explicitly enable the stage even when the safe default config disables it",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.enable:
        config.setdefault("self_training", {})["enabled"] = True
    run_v7_self_training(config, PROJECT_ROOT)


if __name__ == "__main__":
    main()
