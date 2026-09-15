"""Batched, resumable comparisons of frozen model families against bilinear."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .checkpoints import load_model
from .models import build_model
from .prism_data import PrismDataset, audit_manifest
from .prism_metrics import quality_batch
from .prism_training import atomic_json, load_checkpoint
from .research_data import DegradationSpec, load_manifest, sha256_file
from .research_models import load_ablation_checkpoint
from .v7 import load_v7_checkpoint

METRICS = ("chroma_l1", "full_l1", "luma_l1", "chroma_psnr", "chroma_ssim",
           "rgb_psnr", "rgb_ssim", "out_of_range_fraction")
LOWER_IS_BETTER = {"chroma_l1", "full_l1", "luma_l1", "out_of_range_fraction"}


def load_candidate(entry, device):
    kind, path = entry["kind"], entry["weights"]
    if kind == "prism":
        model, checkpoint = load_checkpoint(path, device)
    elif kind == "v7":
        model, checkpoint = load_v7_checkpoint(path, device)
    elif kind == "ablation":
        model, checkpoint = load_ablation_checkpoint(path, device)
    elif kind == "legacy":
        model = build_model(entry["version"]).to(device)
        checkpoint = load_model(model, path, entry["version"], device)
    else:
        raise ValueError(f"Unknown candidate kind: {kind}")
    model.eval()
    return model, {"epoch": checkpoint.get("epoch"),
                   "parameters": sum(p.numel() for p in model.parameters())}


def predict_candidate(model, kind, inputs, specs):
    if kind == "prism":
        return model.predict(inputs, specs).image
    if kind == "v7":
        return model(inputs).ycrcb("mean")
    return model(inputs)


def summarize(values, names, spec_names):
    """Each retained image contributes exactly once to each degradation."""
    results = {}
    for index, name in enumerate(names):
        result = {"metrics": {}, "by_degradation": {}}
        for condition, selected in [("overall", values), *[
            (spec, values[k::len(spec_names)]) for k, spec in enumerate(spec_names)
        ]]:
            metrics = {}
            for column, metric in enumerate(METRICS):
                sample = selected[:, index, column].astype(np.float64)
                baseline = selected[:, 0, column].astype(np.float64)
                gain = baseline - sample if metric in LOWER_IS_BETTER else sample - baseline
                metrics[metric] = {"mean": float(sample.mean()),
                                   "mean_gain_vs_bilinear": float(gain.mean()),
                                   "win_rate_vs_bilinear": float((gain > 0).mean())}
            if condition == "overall":
                result["metrics"] = metrics
            else:
                result["by_degradation"][condition] = metrics
        results[name] = result
    return results


def compare(campaign_path, dataset_root, output, group=0, device="cuda",
            batch_size=16, workers=6, limit=0):
    campaign_path, output = Path(campaign_path), Path(output)
    campaign = json.loads(campaign_path.read_text())
    selected_entries = [row for row in campaign["models"] if group < 0 or row["group"] == group]
    if not selected_entries:
        raise ValueError("No models selected")
    names = ["bilinear", *[row["name"] for row in selected_entries]]
    if len(names) != len(set(names)):
        raise ValueError("Candidate names must be unique")
    suite = campaign["suite"]
    records = load_manifest(campaign["manifest"])
    if any(row.split != "test" for row in records):
        raise ValueError("Comparison requires a test-only manifest")
    records = records[:limit] if limit else records
    audit_manifest(records, dataset_root, suite["data_kind"], suite["training"]["crop_size"])
    specs = [DegradationSpec(**row) for row in suite["degradations"]]
    if any(spec.upsample_filter != "bilinear" for spec in specs):
        raise ValueError("Comparison inputs must use bilinear reconstruction")
    dataset = PrismDataset(records, dataset_root, suite["training"]["crop_size"],
                           specs, suite["training"]["seed"], all_degradations=True)
    fingerprint = {"campaign_sha256": sha256_file(campaign_path), "group": group,
                   "images": len(records), "samples": len(dataset), "methods": names,
                   "metrics": list(METRICS), "batch_size": batch_size,
                   "manifest_sha256": sha256_file(campaign["manifest"])}
    output.mkdir(parents=True, exist_ok=True)
    progress_path, values_path = output / "progress.json", output / "metrics.npy"
    completed = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["fingerprint"] != fingerprint:
            raise ValueError("Resume configuration differs from saved comparison")
        completed = progress["completed_samples"]
        values = np.lib.format.open_memmap(values_path, mode="r+")
    else:
        if values_path.exists():
            raise ValueError("Metrics exist without a progress record")
        values = np.lib.format.open_memmap(values_path, mode="w+", dtype=np.float32,
                                          shape=(len(dataset), len(names), len(METRICS)))
        values.flush()
        atomic_json(progress_path, {"fingerprint": fingerprint, "completed_samples": 0})
    if values.shape != (len(dataset), len(names), len(METRICS)):
        raise ValueError("Invalid saved metric array shape")
    loaded, metadata = [], []
    for entry in selected_entries:
        if sha256_file(entry["weights"]) != entry["sha256"]:
            raise ValueError(f"Checkpoint changed: {entry['name']}")
        model, info = load_candidate(entry, device)
        loaded.append((entry, model))
        metadata.append({**entry, **info})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    loader = DataLoader(Subset(dataset, range(completed, len(dataset))),
                        batch_size=batch_size, num_workers=workers, shuffle=False,
                        pin_memory=str(device).startswith("cuda"), persistent_workers=workers > 0)
    start, initial = time.monotonic(), completed
    method_seconds = {name: 0.0 for name in names}
    print(json.dumps({"status": "evaluating", **fingerprint, "models": metadata}), flush=True)
    with torch.inference_mode():
        for inputs, target, _, spec_indices, _ in loader:
            inputs, target = inputs.to(device), target.to(device)
            batch_specs = [specs[int(index)] for index in spec_indices]
            end = completed + len(inputs)
            for column in range(len(names)):
                method_start = time.monotonic()
                candidate = inputs if column == 0 else predict_candidate(
                    loaded[column - 1][1], loaded[column - 1][0]["kind"], inputs, batch_specs)
                metrics = quality_batch(target, candidate)
                tensor = torch.stack([metrics[key] for key in METRICS], dim=1)
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Nonfinite metrics for {names[column]} at sample {completed}")
                values[completed:end, column] = tensor.cpu().numpy()
                method_seconds[names[column]] += time.monotonic() - method_start
            completed = end
            if completed % (batch_size * 16) == 0 or completed == len(dataset):
                values.flush()
                elapsed = time.monotonic() - start
                progress = {"fingerprint": fingerprint, "completed_samples": completed,
                            "elapsed_this_attempt_seconds": elapsed,
                            "samples_per_second": (completed - initial) / max(elapsed, 1e-6)}
                atomic_json(progress_path, progress)
                print(json.dumps(progress), flush=True)
    report = {"status": "complete", **fingerprint, "baseline": "bilinear",
              "models": metadata, "results": summarize(values, names, [s.name for s in specs]),
              "protocol": {"crop_size": suite["training"]["crop_size"],
                           "degradations": suite["degradations"], "crop": "center",
                           "prediction": "mean", "precision": "float32/TF32",
                           "metric_implementation": "chroma.prism_metrics.quality_batch",
                           "final_clipping": [0, 1]},
              "method_seconds_this_attempt": method_seconds,
              "samples_this_attempt": completed - initial,
              "elapsed_this_attempt_seconds": time.monotonic() - start}
    atomic_json(output / "report.json", report)
    return report
