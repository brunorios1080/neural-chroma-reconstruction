"""Paired synthetic-degradation evaluation of Prism checkpoints and interpolation."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from .prism_data import PrismDataset, audit_manifest
from .prism_training import load_checkpoint, atomic_json
from .research_baselines import classical_reconstruction
from .research_benchmark import aggregate_records
from .research_data import load_manifest, DegradationSpec, sha256_file
from .research_metrics import reconstruction_metrics
from .v7_uncertainty import risk_coverage_curve, spearman_correlation


def evaluate_prism(
    suite,
    weights,
    manifest,
    dataset_root,
    output_dir,
    device="cpu",
    split="test",
    limit=0,
    bootstrap_samples=1000,
):
    baselines = suite.get("evaluation", {}).get("baselines", ["bilinear", "bicubic", "lanczos3"])
    if (not isinstance(baselines, list) or "bilinear" not in baselines
            or any(method not in {"bilinear", "bicubic", "lanczos3"} for method in baselines)
            or len(baselines) != len(set(baselines))):
        raise ValueError("Evaluation baselines must be unique supported methods including bilinear")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Evaluation output already exists: {destination}")
    if split not in {"test", "validation"}:
        raise ValueError("Evaluation split must be test or validation")
    records = load_manifest(manifest)
    crop = int(suite["training"]["crop_size"])
    audit_manifest(records, dataset_root, suite["data_kind"], crop)
    selected = [record for record in records if record.split == split]
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"No records in {split} split")
    specs = [DegradationSpec(**spec) for spec in suite["degradations"]]
    data = PrismDataset(
        selected,
        dataset_root,
        crop,
        specs,
        suite["training"]["seed"],
        all_degradations=True,
    )
    models, metadata = [], []
    for path in weights:
        model, checkpoint = load_checkpoint(path, device)
        model.eval()
        if any(checkpoint["name"] == row[0] for row in models):
            raise ValueError(
                "Duplicate model names in comparison; evaluate distinct experiments"
            )
        # A test source must not have been used for training or checkpoint selection.
        training_manifest = Path(path).parent / "manifest.jsonl"
        if split == "test":
            if not training_manifest.is_file():
                raise ValueError(f"Missing training manifest beside checkpoint: {path}")
            used = {
                r.sha256
                for r in load_manifest(training_manifest)
                if r.split in {"train", "validation"}
            }
            if used.intersection(r.sha256 for r in selected):
                raise ValueError(
                    f"Test data overlaps training/validation for {checkpoint['name']}"
                )
        models.append((checkpoint["name"], model, checkpoint))
        metadata.append(
            {
                "name": checkpoint["name"],
                "weights": str(path),
                "sha256": sha256_file(path),
                "epoch": checkpoint["epoch"],
                "architecture": asdict(model.architecture),
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        )
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode(), (destination / "per_image.jsonl").open("w") as output:
        for sample_index in range(len(data)):
            inputs, target, low, spec_index, identifier = data[sample_index]
            spec = specs[spec_index]
            truth = target.permute(1, 2, 0).numpy()
            tensor = inputs.unsqueeze(0).to(device)
            candidates = []
            for method in baselines:
                chroma = classical_reconstruction(
                    method, truth[..., 0], low.permute(1, 2, 0).numpy(), spec.siting
                )
                candidates.append(
                    (method, np.concatenate((truth[..., :1], chroma), axis=2), {})
                )
            for name, model, checkpoint in models:
                prediction = model.predict(tensor, [spec])
                candidate = prediction.image[0].cpu().permute(1, 2, 0).numpy()
                diagnostics = {}
                if prediction.scale is not None:
                    active = checkpoint["epoch"] > int(
                        checkpoint["config"]["loss"].get("uncertainty_warmup_epochs", 5)
                    )
                    diagnostics["uncertainty_training_started"] = active
                    if active:
                        confidence = prediction.confidence()[0, 0].cpu().numpy()
                        errors = np.mean(
                            np.abs(candidate[..., 1:3].clip(0, 1) - truth[..., 1:3]),
                            axis=2,
                        )
                        correlation = spearman_correlation(1 - confidence, errors)
                        diagnostics.update(
                            risk_coverage=risk_coverage_curve(errors, confidence),
                            uncertainty_error_spearman=(
                                correlation if np.isfinite(correlation) else None
                            ),
                        )
                        amplitude = np.linalg.norm(truth[..., 1:3] - 0.5, axis=2)
                        predicted_a = prediction.amplitude[0, 0].cpu().numpy()
                        scale = prediction.scale[0, 0].cpu().numpy()
                        half = -scale * np.log(0.1)
                        diagnostics["amplitude_coverage_90"] = float(
                            np.mean(np.abs(amplitude - predicted_a) <= half)
                        )
                        diagnostics["amplitude_interval_width_90"] = float(
                            np.mean(2 * half)
                        )
                        safe = prediction.safe_image()[0].cpu().permute(1, 2, 0).numpy()
                        candidates.append(
                            (
                                name + "_safe",
                                safe,
                                {"confidence_source": "prediction_only"},
                            )
                        )
                candidates.append((name, candidate, diagnostics))
            for name, candidate, diagnostics in candidates:
                row = {
                    "image_id": identifier,
                    "degradation_id": spec.name,
                    "degradation": asdict(spec),
                    "method": name,
                    "metrics": reconstruction_metrics(truth, candidate.clip(0, 1)),
                    "profile": {},
                    "diagnostics": diagnostics,
                }
                rows.append(row)
                output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    aggregate, paired = aggregate_records(
        rows, bootstrap_samples, int(suite["training"]["seed"])
    )
    report = {
        "baseline": "bilinear",
        "baseline_methods": baselines,
        "data_kind": suite["data_kind"],
        "split": split,
        "images": len(selected),
        "records": len(rows),
        "manifest_sha256": sha256_file(manifest),
        "models": metadata,
        "aggregate": aggregate,
        "paired_vs_bilinear": paired,
        "note": "Synthetic 4:2:0 evaluation; COCO JPEG sources are not evidence of original lossless chroma recovery. Intervals are nominal until empirical coverage is assessed.",
    }
    atomic_json(destination / "report.json", report)
    atomic_json(destination / "config.snapshot.json", suite)
    return report
