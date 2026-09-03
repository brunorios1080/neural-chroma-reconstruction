#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
# Respect the active virtual/Conda environment. On macOS, `python3` may resolve
# to the dependency-free Xcode interpreter even while `python` is the selected
# project environment.
exec python -m unittest discover -s tests -v
