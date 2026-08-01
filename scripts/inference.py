#!/usr/bin/env python3
"""Run V5 or V6 chroma reconstruction on one image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.checkpoints import load_model
from chroma.data import (
    read_rgb,
    rgb_to_ycrcb,
    simulate_420,
    to_tensor,
    ycrcb_to_bgr_uint8,
)
from chroma.inference import predict
from chroma.models import build_model
from chroma.training import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("v5", "v6"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--comparison", type=Path, help="Optional baseline | AI comparison image"
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = build_model(args.model).to(device)
    checkpoint = load_model(model, args.weights, args.model, device)
    model.eval()

    target = rgb_to_ycrcb(read_rgb(args.image))
    model_input = simulate_420(target)
    input_tensor = to_tensor(model_input).unsqueeze(0).to(device)
    prediction = (
        predict(model, input_tensor, args.model)[0]
        .float()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), ycrcb_to_bgr_uint8(prediction)):
        raise OSError(f"Could not write output image: {args.output}")
    if args.comparison:
        args.comparison.parent.mkdir(parents=True, exist_ok=True)
        comparison = np.hstack(
            [ycrcb_to_bgr_uint8(model_input), ycrcb_to_bgr_uint8(prediction)]
        )
        if not cv2.imwrite(str(args.comparison), comparison):
            raise OSError(f"Could not write comparison image: {args.comparison}")
    print(
        f"Saved {args.model.upper()} output to {args.output} "
        f"(checkpoint epoch {checkpoint.get('epoch', 'unknown')})"
    )


if __name__ == "__main__":
    main()
