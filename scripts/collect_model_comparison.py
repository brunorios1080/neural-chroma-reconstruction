#!/usr/bin/env python3
"""Combine complete groups from a frozen comparison campaign."""
import argparse
import csv
import io
import json
import hashlib
from pathlib import Path
import tempfile


def atomic_text(path, content):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        output.write(content)
        temporary = Path(output.name)
    temporary.replace(path)


def collect(root):
    campaign = json.loads((root / "campaign.json").read_text())
    campaign_hash = hashlib.sha256((root / "campaign.json").read_bytes()).hexdigest()
    groups = sorted({entry["group"] for entry in campaign["models"]})
    reports = [root / "groups" / str(group) / "report.json" for group in groups]
    if not all(path.exists() for path in reports):
        return False
    results, model_info = {}, {}
    for path in reports:
        report = json.loads(path.read_text())
        if (report["status"] != "complete" or report["images"] != campaign["images"]
                or report["campaign_sha256"] != campaign_hash):
            raise ValueError(f"Incomplete group report: {path}")
        for model in report["models"]:
            if model["name"] in model_info:
                raise ValueError("A model occurs in more than one group")
            model_info[model["name"]] = model
        for name, result in report["results"].items():
            results.setdefault(name, result)
    expected = {entry["name"] for entry in campaign["models"]}
    if set(model_info) != expected:
        raise ValueError("Group reports do not cover all campaign models")
    final = {"status": "complete", "images": campaign["images"], "baseline": "bilinear",
             "models": model_info, "results": results, "group_reports": [str(p) for p in reports]}
    atomic_text(root / "comparison.json", json.dumps(final, indent=2) + "\n")
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["model", "training_status", "epoch", "chroma_psnr", "chroma_psnr_gain_db",
                     "chroma_ssim", "rgb_psnr", "rgb_ssim", "chroma_l1", "overlap_audit"])
    for name in ["bilinear", *[entry["name"] for entry in campaign["models"]]]:
        row = results[name]["metrics"]
        model = model_info.get(name, {})
        writer.writerow([name, model.get("training_status", "baseline"), model.get("epoch", ""),
                         row["chroma_psnr"]["mean"], row["chroma_psnr"]["mean_gain_vs_bilinear"],
                         row["chroma_ssim"]["mean"], row["rgb_psnr"]["mean"], row["rgb_ssim"]["mean"],
                         row["chroma_l1"]["mean"], model.get("overlap_audit", "")])
    atomic_text(root / "comparison.csv", stream.getvalue())
    print(f"Complete comparison: {root / 'comparison.csv'}", flush=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    collect(args.campaign)
