#!/usr/bin/env python3
"""Read core-run validation histories without loading models or using a GPU."""

import argparse
import hashlib
import json
from pathlib import Path

CORE = ("prism_residual", "prism_polar", "prism_polar_prob", "prism_cartesian_prob")


def review(root, target_epoch=25):
    results, manifests = [], set()
    for name in CORE:
        directory = Path(root) / name
        path = directory / "history.jsonl"
        # Ignore an incomplete final append while training is running.
        lines = path.read_text().splitlines(keepends=True) if path.exists() else []
        rows = [json.loads(line) for line in lines if line.endswith("\n")]
        if not rows:
            results.append({"model": name, "status": "awaiting metrics"})
            continue
        latest = rows[-1]
        best = min(rows, key=lambda row: row["validation"]["chroma_l1"])
        recent = rows[-5:]
        manifest = directory / "manifest.jsonl"
        digest = (
            hashlib.sha256(manifest.read_bytes()).hexdigest()
            if manifest.exists()
            else None
        )
        manifests.add(digest)
        results.append(
            {
                "model": name,
                "status": (
                    "review ready" if latest["epoch"] >= target_epoch else "in progress"
                ),
                "epoch": latest["epoch"],
                "planned_epochs": latest["epochs"],
                "best_epoch": best["epoch"],
                "best_chroma_l1": best["validation"]["chroma_l1"],
                "latest_chroma_l1": latest["validation"]["chroma_l1"],
                "recent_mean_chroma_l1": sum(
                    r["validation"]["chroma_l1"] for r in recent
                )
                / len(recent),
                "latest_rgb_psnr": latest["validation"]["rgb_psnr"],
                "latest_rgb_ssim": latest["validation"]["rgb_ssim"],
                "latest_chroma_psnr_gain_db": latest["validation"][
                    "chroma_psnr_gain_db"
                ],
            }
        )
    ready = all(r["status"] == "review ready" for r in results)
    matched = ready and len(manifests) == 1 and None not in manifests
    return {
        "target_epoch": target_epoch,
        "ready_for_comparison": bool(matched),
        "runs": results,
        "note": "Validation only; remaining full variants stay unsubmitted pending curve and per-degradation review.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-epoch", type=int, default=25)
    args = parser.parse_args()
    if args.target_epoch < 1:
        parser.error("--target-epoch must be positive")
    print(
        json.dumps(
            review(args.output_root, args.target_epoch), indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
