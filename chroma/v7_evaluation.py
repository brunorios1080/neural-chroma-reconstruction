"""Manifest-driven V7 reconstruction and uncertainty evaluation."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from .chroma_polar import (
    cartesian_chroma_to_polar,
    circular_difference,
    maximum_amplitude_for_phase,
)
from .research_baselines import load_legacy_predictor
from .research_benchmark import resolve_device
from .research_data import (
    DegradationSpec,
    deterministic_crop,
    load_manifest,
    read_rgb,
    records_for_split,
    rgb_to_ycrcb,
    simulate_420,
    verify_manifest,
    ycrcb_to_rgb,
)
from .research_metrics import delta_e_ciede2000, reconstruction_metrics
from .v7 import load_v7_checkpoint, parameter_count
from .v7_losses import degrade_chroma_torch
from .v7_uncertainty import (
    SUPPORTED_COVERAGES,
    VonMisesIntervalLookup,
    circular_variance,
    empirical_interval_coverage,
    laplace_interval,
    spearman_correlation,
)


def _resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(image))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _target_edge_mask(target_chroma: np.ndarray) -> np.ndarray:
    dy = np.gradient(target_chroma.astype(np.float64), axis=0)
    dx = np.gradient(target_chroma.astype(np.float64), axis=1)
    strength = np.mean(np.hypot(dx, dy), axis=2)
    return strength >= np.percentile(strength, 75.0)


def retained_metrics(
    target: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    edge_mask: np.ndarray,
    psnr_cap: float = 80.0,
) -> dict[str, float | None]:
    """Metrics meaningful on a non-spatial confidence-selected pixel subset."""
    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return {
            "chroma_mae": None,
            "chroma_psnr": None,
            "delta_e2000_mean": None,
            "chroma_edge_mae": None,
        }
    chroma_difference = target[..., 1:3] - candidate[..., 1:3]
    chroma_absolute = np.mean(np.abs(chroma_difference), axis=2)
    mse = float(np.mean(np.square(chroma_difference[selected])))
    chroma_psnr = (
        psnr_cap if mse <= 1e-15 else min(psnr_cap, 10.0 * math.log10(1.0 / mse))
    )
    delta_e = delta_e_ciede2000(ycrcb_to_rgb(target), ycrcb_to_rgb(candidate))
    selected_edges = selected & edge_mask
    return {
        "chroma_mae": float(np.mean(chroma_absolute[selected])),
        "chroma_psnr": float(chroma_psnr),
        "delta_e2000_mean": float(np.mean(delta_e[selected])),
        "chroma_edge_mae": (
            float(np.mean(chroma_absolute[selected_edges]))
            if np.any(selected_edges)
            else None
        ),
    }


def detailed_risk_coverage(
    target: np.ndarray,
    candidate: np.ndarray,
    confidence: np.ndarray,
    coverages: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.0),
) -> list[dict[str, Any]]:
    flat_confidence = confidence.ravel()
    order = np.argsort(-flat_confidence, kind="mergesort")
    edge_mask = _target_edge_mask(target[..., 1:3])
    curve = []
    for requested in coverages:
        count = max(1, math.ceil(requested * flat_confidence.size))
        retained = np.zeros(flat_confidence.size, dtype=bool)
        retained[order[:count]] = True
        curve.append(
            {
                "requested_coverage": requested,
                "actual_coverage": float(count / flat_confidence.size),
                "confidence_threshold": float(np.min(flat_confidence[order[:count]])),
                **retained_metrics(
                    target,
                    candidate,
                    retained.reshape(confidence.shape),
                    edge_mask,
                ),
            }
        )
    return curve


def uncertainty_error_bins(
    uncertainty: np.ndarray, error: np.ndarray, bins: int = 10
) -> list[dict[str, float | int]]:
    uncertainty = np.asarray(uncertainty, dtype=np.float64).ravel()
    error = np.asarray(error, dtype=np.float64).ravel()
    order = np.argsort(uncertainty, kind="mergesort")
    groups = np.array_split(order, bins)
    return [
        {
            "bin": index,
            "pixels": int(group.size),
            "mean_uncertainty": float(np.mean(uncertainty[group])),
            "mean_error": float(np.mean(error[group])),
        }
        for index, group in enumerate(groups)
        if group.size
    ]


def _heatmap(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    high = float(np.percentile(finite, 99.0)) if finite.size else 1.0
    normalized = np.clip(values / max(high, 1e-12), 0.0, 1.0)
    red = np.clip(2.0 * normalized, 0.0, 1.0)
    green = np.clip(2.0 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - normalized), 0.0, 1.0)
    return np.stack((red, green, blue), axis=2).astype(np.float32)


def _gradient_peak(target: np.ndarray) -> tuple[int, int]:
    chroma = target[..., 1:3].astype(np.float64)
    strength = np.mean(
        np.hypot(np.gradient(chroma, axis=1), np.gradient(chroma, axis=0)), axis=2
    )
    return tuple(int(v) for v in np.unravel_index(np.argmax(strength), strength.shape))


def _crop_bounds(
    center: tuple[int, int], height: int, width: int, size: int
) -> tuple[slice, slice, dict[str, int]]:
    crop_height, crop_width = min(size, height), min(size, width)
    top = min(max(0, center[0] - crop_height // 2), height - crop_height)
    left = min(max(0, center[1] - crop_width // 2), width - crop_width)
    return (
        slice(top, top + crop_height),
        slice(left, left + crop_width),
        {"top": top, "left": left, "height": crop_height, "width": crop_width},
    )


def save_v7_diagnostic_sheet(
    path: Path,
    target: np.ndarray,
    model_input: np.ndarray,
    mean: np.ndarray,
    safe: np.ndarray,
    amplitude_uncertainty: np.ndarray,
    phase_uncertainty: np.ndarray,
    crop_size: int,
    v6: np.ndarray | None = None,
) -> dict[str, int]:
    """Save difficult-boundary reconstruction and uncertainty panels."""
    rows, columns, bounds = _crop_bounds(
        _gradient_peak(target), target.shape[0], target.shape[1], crop_size
    )
    chroma_error = np.mean(np.abs(target[..., 1:3] - mean[..., 1:3]), axis=2)
    panels: list[tuple[str, np.ndarray]] = [
        ("reference", ycrcb_to_rgb(target)),
        ("bilinear", ycrcb_to_rgb(model_input)),
    ]
    if v6 is not None:
        panels.append(("V6", ycrcb_to_rgb(v6)))
    panels.extend(
        [
            ("V7 mean", ycrcb_to_rgb(mean)),
            ("V7 safe", ycrcb_to_rgb(safe)),
            ("chroma error", _heatmap(chroma_error)),
            ("amplitude scale", _heatmap(amplitude_uncertainty)),
            ("phase circ. variance", _heatmap(phase_uncertainty)),
        ]
    )
    rendered = []
    for label, panel in panels:
        pixels = np.rint(np.clip(panel[rows, columns], 0.0, 1.0) * 255).astype(np.uint8)
        image = Image.fromarray(pixels, mode="RGB")
        scale = max(1, 192 // max(image.size))
        image = image.resize(
            (image.width * scale, image.height * scale), Image.Resampling.NEAREST
        )
        canvas = Image.new("RGB", (image.width, image.height + 22), (24, 24, 24))
        canvas.paste(image, (0, 22))
        ImageDraw.Draw(canvas).text((5, 5), label, fill=(255, 255, 255))
        rendered.append(canvas)
    sheet = Image.new(
        "RGB",
        (
            sum(panel.width for panel in rendered),
            max(panel.height for panel in rendered),
        ),
        (255, 255, 255),
    )
    offset = 0
    for panel in rendered:
        sheet.paste(panel, (offset, 0))
        offset += panel.width
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return bounds


def save_chroma_plane(
    path: Path, candidates: Mapping[str, np.ndarray], neutral_chroma: float
) -> None:
    size, margin = 512, 24
    image = Image.new("RGB", (size, size), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    center = margin + neutral_chroma * (size - 2 * margin)
    draw.line((center, margin, center, size - margin), fill=(160, 160, 160))
    draw.line(
        (margin, size - center, size - margin, size - center), fill=(160, 160, 160)
    )
    colors = [(30, 30, 30), (70, 120, 220), (220, 70, 70), (60, 160, 80)]
    for (label, ycrcb), color in zip(candidates.items(), colors):
        chroma = ycrcb[..., 1:3].reshape(-1, 2)[
            :: max(1, ycrcb.shape[0] * ycrcb.shape[1] // 2500)
        ]
        x = margin + chroma[:, 0] * (size - 2 * margin)
        y = size - margin - chroma[:, 1] * (size - 2 * margin)
        for px, py in zip(x, y):
            draw.point((float(px), float(py)), fill=color)
        draw.text((margin, 4 + 14 * list(candidates).index(label)), label, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _prediction_diagnostics(
    target_tensor: torch.Tensor,
    prediction,
    spec: DegradationSpec,
    observed_low: torch.Tensor,
    phase_lookups: Mapping[float, VonMisesIntervalLookup],
    phase_reference_amplitude: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    target_amplitude, target_phase = cartesian_chroma_to_polar(
        target_tensor[:, 1:3], prediction.neutral_chroma
    )
    amplitude_error = (target_amplitude - prediction.amplitude_mean).abs()
    phase_error = circular_difference(target_phase, prediction.phase_mean).abs()
    target_weight = target_amplitude
    phase_variance = circular_variance(prediction.phase_kappa)
    chroma_error = torch.mean(
        torch.abs(target_tensor[:, 1:3] - prediction.chroma_mean), dim=1, keepdim=True
    )
    phase_relevance = (target_amplitude / phase_reference_amplitude).clamp(0.0, 1.0)
    confidence = prediction.amplitude_confidence * (
        (1.0 - phase_relevance) + phase_relevance * prediction.phase_confidence
    )
    forward = torch.mean(
        torch.abs(degrade_chroma_torch(prediction.chroma_mean, spec) - observed_low)
    )
    arrays = {
        "chroma_absolute_error": chroma_error[0, 0].detach().float().cpu().numpy(),
        "amplitude_absolute_error": amplitude_error[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy(),
        "phase_absolute_error_radians": phase_error[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy(),
        "amplitude_scale": prediction.amplitude_scale[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy(),
        "phase_kappa": prediction.phase_kappa[0, 0].detach().float().cpu().numpy(),
        "phase_circular_variance": phase_variance[0, 0].detach().float().cpu().numpy(),
        "amplitude_confidence": prediction.amplitude_confidence[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy(),
        "phase_confidence": prediction.phase_confidence[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy(),
        "combined_confidence": confidence[0, 0].detach().float().cpu().numpy(),
        "target_amplitude": target_amplitude[0, 0].detach().float().cpu().numpy(),
        "amplitude_mean": prediction.amplitude_mean[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy(),
        "phase_mean_radians": prediction.phase_mean[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy(),
    }
    intervals: dict[str, Any] = {}
    radial_limit = maximum_amplitude_for_phase(
        prediction.phase_mean,
        prediction.neutral_chroma,
    )
    for coverage in SUPPORTED_COVERAGES:
        label = str(round(coverage * 100))
        lower, upper = laplace_interval(
            prediction.amplitude_mean,
            prediction.amplitude_scale,
            coverage,
            upper_bound=radial_limit,
        )
        half_width = phase_lookups[coverage].half_width(prediction.phase_kappa)
        intervals[label] = {
            "amplitude_empirical_coverage": empirical_interval_coverage(
                arrays["target_amplitude"],
                lower[0, 0].detach().cpu().numpy(),
                upper[0, 0].detach().cpu().numpy(),
            ),
            "amplitude_mean_width": float(torch.mean(upper - lower)),
            "phase_empirical_coverage": float(
                torch.mean((phase_error <= half_width).float())
            ),
            "phase_mean_half_width_radians_approx": float(torch.mean(half_width)),
        }
        if coverage == 0.90:
            arrays["amplitude_lower_90"] = lower[0, 0].detach().float().cpu().numpy()
            arrays["amplitude_upper_90"] = upper[0, 0].detach().float().cpu().numpy()
            arrays["phase_half_width_90_radians_approx"] = (
                half_width[0, 0].detach().float().cpu().numpy()
            )
    amplitude_denominator = target_weight.sum().clamp_min(1e-8)
    diagnostics = {
        "amplitude_mae": float(amplitude_error.mean()),
        "phase_error_radians": float(phase_error.mean()),
        "phase_error_degrees": float(phase_error.mean() * (180.0 / math.pi)),
        "phase_error_target_amplitude_weighted_radians": float(
            (phase_error * target_weight).sum() / amplitude_denominator
        ),
        "amplitude_scale_mean": float(prediction.amplitude_scale.mean()),
        "phase_kappa_mean": float(prediction.phase_kappa.mean()),
        "phase_kappa_median": float(prediction.phase_kappa.median()),
        "phase_kappa_p05": float(np.percentile(arrays["phase_kappa"], 5.0)),
        "phase_kappa_p95": float(np.percentile(arrays["phase_kappa"], 95.0)),
        "phase_circular_variance_mean": float(phase_variance.mean()),
        "forward_consistency_l1": float(forward),
        "spearman_amplitude_scale_vs_amplitude_error": _finite_or_none(
            spearman_correlation(
                arrays["amplitude_scale"], arrays["amplitude_absolute_error"]
            )
        ),
        "spearman_phase_variance_vs_phase_error": _finite_or_none(
            spearman_correlation(
                arrays["phase_circular_variance"],
                arrays["phase_absolute_error_radians"],
            )
        ),
        "spearman_uncertainty_vs_chroma_error": _finite_or_none(
            spearman_correlation(
                1.0 - arrays["combined_confidence"], arrays["chroma_absolute_error"]
            )
        ),
        "amplitude_intervals": intervals,
        "phase_interval_method": "numerical symmetric von Mises CDF lookup (approximate)",
        "uncertainty_error_bins": uncertainty_error_bins(
            1.0 - arrays["combined_confidence"], arrays["chroma_absolute_error"]
        ),
    }
    return diagnostics, arrays


def run_v7_evaluation(
    configuration: Mapping[str, Any], project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest_path = _resolve(str(configuration["manifest"]), root)
    dataset_root = _resolve(str(configuration["dataset_root"]), root)
    output_dir = _resolve(str(configuration["output_dir"]), root)
    checkpoint_path = _resolve(str(configuration["weights"]), root)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(manifest_path)
    verification = verify_manifest(records, dataset_root)
    if not verification["ok"]:
        raise RuntimeError(f"Manifest verification failed: {verification}")
    selected = records_for_split(records, str(configuration.get("split", "test")))
    limit = int(configuration.get("limit", 0))
    if limit:
        selected = selected[:limit]
    device = resolve_device(str(configuration.get("device", "auto")))
    model, checkpoint = load_v7_checkpoint(checkpoint_path, device)
    model.eval()
    v6_predictor = None
    v6_weights = configuration.get("v6_weights")
    if v6_weights:
        _, v6_predictor, _ = load_legacy_predictor(
            "v6", _resolve(str(v6_weights), root), device
        )
    specs = [DegradationSpec(**entry) for entry in configuration["degradations"]]
    for spec in specs:
        spec.validate()
    phase_lookups = {
        coverage: VonMisesIntervalLookup.build(
            coverage, kappa_max=model.config.kappa_max
        )
        for coverage in SUPPORTED_COVERAGES
    }
    per_image: list[dict[str, Any]] = []
    qualitative: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qualitative_count = int(configuration.get("qualitative_count", 4))
    pixel_dir = output_dir / "pixel_maps"
    if bool(configuration.get("save_pixel_arrays", True)):
        pixel_dir.mkdir(parents=True, exist_ok=True)
    seed = int(configuration.get("seed", 2026))
    crop_size = int(configuration.get("crop_size", 256))
    for index, record in enumerate(selected, start=1):
        rgb, crop = deterministic_crop(
            read_rgb(dataset_root / record.relative_path), record.id, crop_size, seed
        )
        target = rgb_to_ycrcb(rgb)
        target_tensor = _to_tensor(target, device)
        print(f"[{index}/{len(selected)}] {record.id}")
        for spec in specs:
            model_input, low = simulate_420(target, spec)
            input_tensor = _to_tensor(model_input, device)
            observed_low = _to_tensor(low, device)[:, 0:2]
            with torch.inference_mode():
                prediction = model(input_tensor)
            mean = (
                prediction.ycrcb("mean")[0]
                .detach()
                .float()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            safe = (
                prediction.ycrcb("safe")[0]
                .detach()
                .float()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            diagnostics, arrays = _prediction_diagnostics(
                target_tensor,
                prediction,
                spec,
                observed_low,
                phase_lookups,
                float(checkpoint["loss"].get("phase_reference_amplitude", 0.05)),
            )
            if bool(configuration.get("save_pixel_arrays", True)):
                map_path = (
                    pixel_dir / f"{_safe_name(spec.name)}__{_safe_name(record.id)}.npz"
                )
                np.savez_compressed(map_path, **arrays)
                pixel_path = map_path.relative_to(output_dir).as_posix()
            else:
                pixel_path = None
            for mode, candidate in (("v7_mean", mean), ("v7_safe", safe)):
                metrics = reconstruction_metrics(target, candidate)
                per_image.append(
                    {
                        "image_id": record.id,
                        "sha256": record.sha256,
                        "relative_path": record.relative_path,
                        "crop": crop,
                        "degradation_id": spec.name,
                        "degradation": asdict(spec),
                        "method": mode,
                        "metrics": metrics,
                        "v7": diagnostics,
                        "risk_coverage": detailed_risk_coverage(
                            target, candidate, arrays["combined_confidence"]
                        ),
                        "pixel_maps": pixel_path,
                    }
                )
            v6 = v6_predictor(model_input) if v6_predictor is not None else None
            difficulty = reconstruction_metrics(target, model_input)["chroma_edge_mae"]
            qualitative[spec.name].append(
                {
                    "difficulty": difficulty,
                    "image_id": record.id,
                    "target": target,
                    "model_input": model_input,
                    "mean": mean,
                    "safe": safe,
                    "v6": v6,
                    "amplitude_uncertainty": arrays["amplitude_scale"],
                    "phase_uncertainty": arrays["phase_circular_variance"],
                }
            )
            qualitative[spec.name] = sorted(
                qualitative[spec.name],
                key=lambda item: item["difficulty"],
                reverse=True,
            )[:qualitative_count]
    per_image_path = output_dir / "per_image.jsonl"
    per_image_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in per_image
        ),
        encoding="utf-8",
    )
    aggregate: dict[str, Any] = defaultdict(dict)
    metric_names = list(next(iter(per_image))["metrics"]) if per_image else []
    for spec in specs:
        for method in ("v7_mean", "v7_safe"):
            rows = [
                row
                for row in per_image
                if row["degradation_id"] == spec.name and row["method"] == method
            ]
            if rows:
                method_aggregate: dict[str, Any] = {
                    name: _summary([float(row["metrics"][name]) for row in rows])
                    for name in metric_names
                }
                v7_scalar_names = [
                    name
                    for name, value in rows[0]["v7"].items()
                    if isinstance(value, (int, float)) and value is not None
                ]
                method_aggregate["v7"] = {
                    name: _summary(
                        [
                            float(row["v7"][name])
                            for row in rows
                            if row["v7"].get(name) is not None
                        ]
                    )
                    for name in v7_scalar_names
                }
                method_aggregate["amplitude_intervals"] = {
                    label: {
                        field: _summary(
                            [
                                float(row["v7"]["amplitude_intervals"][label][field])
                                for row in rows
                            ]
                        )
                        for field in rows[0]["v7"]["amplitude_intervals"][label]
                    }
                    for label in rows[0]["v7"]["amplitude_intervals"]
                }
                method_aggregate["risk_coverage"] = [
                    {
                        "requested_coverage": point["requested_coverage"],
                        **{
                            field: _summary(
                                [
                                    float(row["risk_coverage"][point_index][field])
                                    for row in rows
                                    if row["risk_coverage"][point_index][field]
                                    is not None
                                ]
                            )
                            for field in (
                                "actual_coverage",
                                "confidence_threshold",
                                "chroma_mae",
                                "chroma_psnr",
                                "delta_e2000_mean",
                                "chroma_edge_mae",
                            )
                            if any(
                                row["risk_coverage"][point_index][field] is not None
                                for row in rows
                            )
                        },
                    }
                    for point_index, point in enumerate(rows[0]["risk_coverage"])
                ]
                aggregate[spec.name][method] = method_aggregate
    qualitative_index = []
    for degradation, entries in qualitative.items():
        for rank, entry in enumerate(entries, start=1):
            filename = f"{_safe_name(degradation)}_{rank:02d}_{_safe_name(entry['image_id'])}.png"
            bounds = save_v7_diagnostic_sheet(
                output_dir / "qualitative" / filename,
                entry["target"],
                entry["model_input"],
                entry["mean"],
                entry["safe"],
                entry["amplitude_uncertainty"],
                entry["phase_uncertainty"],
                int(configuration.get("qualitative_crop_size", 64)),
                entry["v6"],
            )
            plane_name = filename.replace(".png", "_chroma_plane.png")
            planes = {
                "reference": entry["target"],
                "bilinear": entry["model_input"],
                "V7 mean": entry["mean"],
                "V7 safe": entry["safe"],
            }
            save_chroma_plane(
                output_dir / "qualitative" / plane_name,
                planes,
                model.config.neutral_chroma,
            )
            qualitative_index.append(
                {
                    "degradation_id": degradation,
                    "rank": rank,
                    "image_id": entry["image_id"],
                    "difficulty": entry["difficulty"],
                    "crop": bounds,
                    "file": f"qualitative/{filename}",
                    "chroma_plane": f"qualitative/{plane_name}",
                }
            )
    report = {
        "protocol": "v7-recoverability-evaluation-v1",
        "scientific_claim": "none; outputs require interpretation on held-out data",
        "config": dict(configuration),
        "checkpoint": {
            "path": str(configuration["weights"]),
            "epoch": checkpoint["epoch"],
            "git_commit": checkpoint.get("git_commit"),
            "architecture": checkpoint["architecture"],
            "loss": checkpoint["loss"],
            "parameters": parameter_count(model),
        },
        "manifest_verification": verification,
        "images": len(selected),
        "records": len(per_image),
        "aggregate": dict(aggregate),
        "qualitative_index": qualitative_index,
        "interval_note": "Laplace intervals are exact before physical clipping; von Mises half-widths use a numerical symmetric-CDF lookup.",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.snapshot.json").write_text(
        json.dumps(dict(configuration), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
