#!/usr/bin/env python3
"""Freeze one representative checkpoint per available model variant."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chroma.model_comparison import load_candidate
from chroma.prism_config import load_suite
from chroma.research_data import load_manifest, sha256_file


def snapshot(destination, storage):
    if (destination / "campaign.json").exists():
        raise FileExistsError("A completed campaign must not be overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    checkpoints = storage / "checkpoints"
    suite = load_suite(ROOT / "research/configs/prism_coco_test2014.json")
    core = {"prism_residual", "prism_polar", "prism_polar_prob", "prism_cartesian_prob"}
    selected, smoke_index = [], 0
    for experiment in suite["experiments"]:
        name = experiment["name"]
        trained = name in core
        parent = "prism_coco" if trained else "prism_gpu_smoke_smoke100"
        group = 0 if trained else 1 + smoke_index // 4
        smoke_index += int(not trained)
        selected.append({"name": name, "kind": "prism", "group": group,
                         "source": str(checkpoints / "prism" / parent / name / "best.pth"),
                         "training_status": "full_dataset_checkpoint" if trained else "two_epoch_100_image_smoke",
                         "overlap_audit": "known_training_sources_excluded"})
    for name, kind, version, path in (
        ("v5", "legacy", "v5", ROOT / "models/version5/epoch_010.pth"),
        ("v5_1", "legacy", "v5", checkpoints / "v5.1/best.pth"),
        ("v6", "legacy", "v6", ROOT / "models/version6/res_epoch_030.pth"),
        ("v7", "v7", "v7", checkpoints / "v7/v7_coco/best.pth"),
    ):
        selected.append({"name": name, "kind": kind, "version": version, "group": 3,
                         "source": str(path), "training_status": "historical_checkpoint",
                         "overlap_audit": "original_training_manifest_unavailable"})
    for path in sorted((ROOT / "research/checkpoints/smoke").glob("*/best.pth")):
        selected.append({"name": "ablation_" + path.parent.name, "kind": "ablation", "group": 4,
                         "source": str(path), "training_status": "one_epoch_fixture_smoke",
                         "overlap_audit": "known_training_sources_excluded"})
    # Legacy COCO runs used different splits. Exclude the entire known COCO source
    # manifest rather than only Prism's train/validation partitions.
    exclusion_sources = [checkpoints / "prism/prism_coco/prism_residual/manifest.jsonl",
                         ROOT / "research/manifests/fixture.jsonl"]
    used_hashes, used_ids = set(), set()
    for source in exclusion_sources:
        for row in load_manifest(source):
            used_hashes.add(row.sha256)
            stem = Path(row.relative_path).stem.split("_")[-1]
            if stem.isdigit():
                used_ids.add(int(stem))
    original_manifest = storage / "data/coco/manifests/prism_coco_test2014.jsonl"
    records = load_manifest(original_manifest)
    retained = [row for row in records if row.sha256 not in used_hashes
                and int(Path(row.relative_path).stem.split("_")[-1]) not in used_ids]
    manifest = destination / "test_manifest.jsonl"
    manifest.write_text("".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in retained))
    for entry in selected:
        target = destination / "weights" / entry["name"] / "selected.pth"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(".tmp")
            shutil.copyfile(entry["source"], temporary)
            temporary.replace(target)
        entry.update(weights=str(target), sha256=sha256_file(target))
        model, info = load_candidate(entry, "cpu")
        entry.update(info)
        del model
        print(json.dumps(entry), flush=True)
    payload = {"suite": suite, "manifest": str(manifest), "models": selected,
               "baseline": "bilinear", "original_images": len(records), "images": len(retained),
               "additional_overlap_exclusions": len(records) - len(retained),
               "exclusion_sources": [{"path": str(p), "sha256": sha256_file(p)} for p in exclusion_sources],
               "selection": "Best validation checkpoint for each distinct variant; historical V5/V6 use their sole available checkpoint. Smoke/benchmark duplicates of a full-data model are not additional variants."}
    (destination / "campaign.json").write_text(json.dumps(payload, indent=2) + "\n")
    code = destination / "code"
    for folder in ("chroma", "scripts", "research/configs"):
        shutil.copytree(ROOT / folder, code / folder, ignore=shutil.ignore_patterns("__pycache__"))
    print(f"Frozen {len(selected)} models; {len(retained)} test images; campaign {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage", type=Path, default=Path("/ocean/projects/cis260224p/shared/brios"))
    args = parser.parse_args()
    snapshot(args.output.resolve(), args.storage)
