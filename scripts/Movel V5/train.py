#!/usr/bin/env python3
"""Compatibility wrapper for the former V5 training entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "train.py"
os.execv(sys.executable, [sys.executable, str(SCRIPT), "--model", "v5", *sys.argv[1:]])
