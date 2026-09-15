#!/usr/bin/env bash
# Install the one dependency missing from PSC's PyTorch module.

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
dependency_dir="$project_root/.cluster/python"

if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
fi
module purge
module load pytorch/26.05-2.11-py3

mkdir -p "$dependency_dir"
if ! PYTHONPATH="$dependency_dir${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -c 'import cv2' >/dev/null 2>&1; then
    uv pip install \
        --python "$(command -v python3)" \
        --target "$dependency_dir" \
        --no-deps \
        'opencv-python-headless>=4.8,<5'
fi

PYTHONPATH="$dependency_dir${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import cv2
import numpy
import torch

print(f"Python environment ready: torch={torch.__version__}, "
      f"numpy={numpy.__version__}, cv2={cv2.__version__}")
PY
