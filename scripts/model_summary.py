#!/usr/bin/env python3
"""Print V5, V6, and V7 parameter summaries."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.models import Discriminator, build_model, parameter_count
from chroma.v7 import V7PolarChromaRefiner


def main() -> None:
    v5 = build_model("v5")
    discriminator = Discriminator()
    v6 = build_model("v6")
    v7 = V7PolarChromaRefiner()
    print(f"V5 generator:     {parameter_count(v5):>10,} parameters")
    print(f"V5 discriminator: {parameter_count(discriminator):>10,} parameters")
    print(
        f"V5 total:         {parameter_count(v5) + parameter_count(discriminator):>10,} parameters"
    )
    print(f"V6 refiner:       {parameter_count(v6):>10,} parameters")
    print(f"V7 polar refiner: {parameter_count(v7):>10,} parameters")


if __name__ == "__main__":
    main()
