#!/usr/bin/env python3
"""Run an untrained V7 software smoke evaluation on one procedural fixture."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chroma.research_benchmark import run_benchmark
from chroma.v7 import V7PolarChromaRefiner, save_v7_checkpoint
from chroma.v7_evaluation import run_v7_evaluation
from chroma.v7_losses import V7LossConfig


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="v7_smoke_") as temporary:
        directory = Path(temporary)
        checkpoint = directory / "initialized_v7.pth"
        save_v7_checkpoint(
            checkpoint,
            V7PolarChromaRefiner(),
            epoch=0,
            seed=2026,
            loss_config=asdict(V7LossConfig()),
            git_commit=None,
            extra={"software_smoke_only": True},
        )
        report = run_v7_evaluation(
            {
                "manifest": "research/manifests/fixture.jsonl",
                "dataset_root": "research/fixtures/lossless",
                "weights": str(checkpoint),
                "output_dir": str(directory / "report"),
                "split": "test",
                "crop_size": 64,
                "seed": 2026,
                "device": "cpu",
                "limit": 1,
                "save_pixel_arrays": True,
                "qualitative_count": 1,
                "qualitative_crop_size": 32,
                "degradations": [
                    {
                        "name": "center_box_bilinear",
                        "siting": "center",
                        "downsample_filter": "box",
                        "upsample_filter": "bilinear",
                    }
                ],
            },
            PROJECT_ROOT,
        )
        if report["records"] != 2:
            raise RuntimeError("V7 smoke evaluation did not emit mean and safe rows")
        publication_report = run_benchmark(
            {
                "manifest": "research/manifests/fixture.jsonl",
                "dataset_root": "research/fixtures/lossless",
                "output_dir": str(directory / "publication_report"),
                "split": "test",
                "crop_size": 64,
                "seed": 2026,
                "device": "cpu",
                "limit": 1,
                "profile_warmup": 0,
                "profile_repeats": 1,
                "bootstrap_samples": 10,
                "synthetic_degradations": [
                    {
                        "name": "center_box_bilinear",
                        "siting": "center",
                        "downsample_filter": "box",
                        "upsample_filter": "bilinear",
                    }
                ],
                "classical_methods": ["bilinear"],
                "learned_methods": [
                    {
                        "name": "v7_mean",
                        "type": "v7",
                        "mode": "mean",
                        "weights": str(checkpoint),
                    },
                    {
                        "name": "v7_safe",
                        "type": "v7",
                        "mode": "safe",
                        "weights": str(checkpoint),
                    },
                ],
                "jpeg_qualities": [],
                "video_codecs": [],
                "qualitative_count": 1,
                "qualitative_crop_size": 32,
                "qualitative_methods": ["bilinear", "v7_mean", "v7_safe"],
            },
            PROJECT_ROOT,
        )
        if publication_report["per_image_records"] != 3:
            raise RuntimeError(
                "Publication smoke did not retain bilinear and both V7 modes"
            )
        print(
            "V7 dedicated and publication software smoke passed "
            "(initialized model; no benchmark claim)."
        )


if __name__ == "__main__":
    main()
