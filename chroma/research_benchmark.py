"""Manifest-driven benchmark with paired statistics and qualitative exports."""

from __future__ import annotations

import json
import platform
import subprocess
import time
import tracemalloc
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import psutil
import torch
from PIL import Image, ImageDraw, __version__ as pillow_version

from .research_baselines import (
    CLASSICAL_METHODS,
    classical_reconstruction,
    load_ablation_predictor,
    load_legacy_predictor,
    load_torchscript_predictor,
    load_v7_predictor,
)
from .research_codecs import ffmpeg_capabilities, jpeg_roundtrip, video_roundtrip
from .research_data import (
    DegradationSpec,
    deterministic_crop,
    load_manifest,
    read_rgb,
    records_for_split,
    rgb_to_ycrcb,
    sha256_file,
    simulate_420,
    verify_manifest,
    ycrcb_to_rgb,
)
from .research_metrics import reconstruction_metrics
from .research_models import convolution_flops, parameter_bytes, parameter_count

HIGHER_IS_BETTER = {"rgb_psnr", "chroma_psnr", "rgb_ssim", "chroma_ssim"}
QUALITY_METRICS = (
    "rgb_psnr",
    "chroma_psnr",
    "rgb_ssim",
    "chroma_ssim",
    "delta_e2000_mean",
    "delta_e2000_p95",
    "chroma_mae",
    "chroma_edge_mae",
    "chroma_gradient_mae",
)


@dataclass
class LearnedMethod:
    name: str
    model: torch.nn.Module
    predictor: Callable[[np.ndarray], np.ndarray]
    metadata: dict[str, Any]


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _portable_project_path(path: Path, project_root: Path) -> str:
    """Represent provenance paths without embedding a contributor's home path."""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def profile_callable(
    function: Callable[[], np.ndarray],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[np.ndarray, dict[str, float]]:
    for _ in range(max(0, warmup)):
        function()
        _sync(device)
    timings = []
    output = None
    process = psutil.Process()
    rss_before = process.memory_info().rss
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    mps_before = (
        int(torch.mps.current_allocated_memory())
        if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory")
        else 0
    )
    tracemalloc.start()
    for _ in range(max(1, repeats)):
        _sync(device)
        start = time.perf_counter_ns()
        output = function()
        _sync(device)
        timings.append((time.perf_counter_ns() - start) / 1_000_000.0)
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = process.memory_info().rss
    if device.type == "cuda":
        accelerator_peak = int(torch.cuda.max_memory_allocated(device))
    elif device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        accelerator_peak = max(
            0, int(torch.mps.current_allocated_memory()) - mps_before
        )
    else:
        accelerator_peak = 0
    assert output is not None
    return output, {
        "runtime_ms_mean": float(np.mean(timings)),
        "runtime_ms_median": float(np.median(timings)),
        "runtime_ms_p95": float(np.percentile(timings, 95.0)),
        "python_peak_bytes": float(python_peak),
        "rss_delta_bytes": float(max(0, rss_after - rss_before)),
        "accelerator_peak_bytes": float(accelerator_peak),
    }


def directional_delta(metric: str, candidate: float, reference: float) -> float:
    return candidate - reference if metric in HIGHER_IS_BETTER else reference - candidate


def bootstrap_interval(
    values: np.ndarray, samples: int, generator: np.random.Generator
) -> list[float]:
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    if not len(values):
        raise ValueError("cannot bootstrap an empty sample")
    # Bound the temporary index matrix for large publication datasets.
    batch_size = max(1, min(samples, 2_000_000 // len(values)))
    batches = []
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        indices = generator.integers(0, len(values), size=(count, len(values)))
        batches.append(np.mean(values[indices], axis=1))
    means = np.concatenate(batches)
    low, high = np.percentile(means, (2.5, 97.5))
    return [float(low), float(high)]


def _summaries(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def aggregate_records(
    records: list[dict[str, Any]], bootstrap_samples: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["degradation_id"], record["method"])].append(record)
    aggregate: dict[str, Any] = defaultdict(dict)
    for (degradation, method), method_records in sorted(grouped.items()):
        numeric_fields = list(QUALITY_METRICS) + [
            "runtime_ms_mean",
            "runtime_ms_median",
            "runtime_ms_p95",
            "python_peak_bytes",
            "rss_delta_bytes",
            "accelerator_peak_bytes",
        ]
        aggregate[degradation][method] = {
            field: _summaries(
                [
                    float(record["metrics"].get(field, record["profile"].get(field)))
                    for record in method_records
                ]
            )
            for field in numeric_fields
            if all(
                record["metrics"].get(field, record["profile"].get(field)) is not None
                for record in method_records
            )
        }
        aggregate[degradation][method]["images"] = len(method_records)
        encoded_bytes = [
            record["degradation"].get("encoded_bytes")
            for record in method_records
        ]
        if all(value is not None for value in encoded_bytes):
            aggregate[degradation][method]["encoded_bytes"] = _summaries(
                [float(value) for value in encoded_bytes]
            )

    paired: dict[str, Any] = defaultdict(dict)
    generator = np.random.default_rng(seed)
    degradation_names = sorted({record["degradation_id"] for record in records})
    for degradation in degradation_names:
        degradation_records = [
            record for record in records if record["degradation_id"] == degradation
        ]
        methods = sorted({record["method"] for record in degradation_records})
        baseline = "bilinear" if "bilinear" in methods else "codec_native"
        indexed = {
            (record["image_id"], record["method"]): record
            for record in degradation_records
        }
        image_ids = sorted(
            {
                record["image_id"]
                for record in degradation_records
                if (record["image_id"], baseline) in indexed
            }
        )
        for method in methods:
            if method == baseline:
                continue
            comparisons: dict[str, Any] = {}
            paired_ids = [
                image_id
                for image_id in image_ids
                if (image_id, method) in indexed
            ]
            for metric in QUALITY_METRICS:
                deltas = np.asarray(
                    [
                        directional_delta(
                            metric,
                            indexed[(image_id, method)]["metrics"][metric],
                            indexed[(image_id, baseline)]["metrics"][metric],
                        )
                        for image_id in paired_ids
                    ],
                    dtype=np.float64,
                )
                if not len(deltas):
                    continue
                comparisons[metric] = {
                    "mean_improvement": float(np.mean(deltas)),
                    "median_improvement": float(np.median(deltas)),
                    "bootstrap_95_ci": bootstrap_interval(
                        deltas, bootstrap_samples, generator
                    ),
                    "win_rate": float(np.mean(deltas > 0.0)),
                    "tie_rate": float(np.mean(deltas == 0.0)),
                    "pairs": len(deltas),
                }
            paired[degradation][method] = {
                "reference": baseline,
                "metrics": comparisons,
            }
    return dict(aggregate), dict(paired)


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_learned_methods(
    configurations: list[Mapping[str, Any]],
    project_root: Path,
    device: torch.device,
) -> list[LearnedMethod]:
    methods = []
    for configuration in configurations:
        method_type = str(configuration["type"])
        weights = _resolve_path(str(configuration["weights"]), project_root)
        if method_type == "legacy":
            model, predictor, metadata = load_legacy_predictor(
                str(configuration["version"]), weights, device
            )
        elif method_type == "ablation":
            model, predictor, metadata = load_ablation_predictor(weights, device)
        elif method_type == "torchscript":
            model, predictor, metadata = load_torchscript_predictor(weights, device)
        elif method_type == "v7":
            model, predictor, metadata = load_v7_predictor(
                weights, device, str(configuration.get("mode", "mean"))
            )
        else:
            raise ValueError(f"Unknown learned method type: {method_type}")
        metadata["weights"] = str(configuration["weights"])
        methods.append(
            LearnedMethod(
                name=str(configuration["name"]),
                model=model,
                predictor=predictor,
                metadata=metadata,
            )
        )
    return methods


def _gradient_peak(target: np.ndarray) -> tuple[int, int]:
    chroma = target[..., 1:3].astype(np.float64)
    dx = np.gradient(chroma, axis=1)
    dy = np.gradient(chroma, axis=0)
    strength = np.mean(np.hypot(dx, dy), axis=2)
    return tuple(int(value) for value in np.unravel_index(np.argmax(strength), strength.shape))


def _crop_bounds(
    center: tuple[int, int], height: int, width: int, size: int
) -> tuple[int, int, int, int]:
    crop_height = min(size, height)
    crop_width = min(size, width)
    top = min(max(0, center[0] - crop_height // 2), height - crop_height)
    left = min(max(0, center[1] - crop_width // 2), width - crop_width)
    return top, left, crop_height, crop_width


def save_qualitative_sheet(
    path: Path,
    target: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    center: tuple[int, int],
    crop_size: int,
) -> dict[str, int]:
    top, left, height, width = _crop_bounds(
        center, target.shape[0], target.shape[1], crop_size
    )
    panels = {"original": target, **candidates}
    rendered = []
    label_height = 22
    scale = max(1, 192 // max(height, width))
    for label, ycrcb in panels.items():
        rgb = ycrcb_to_rgb(ycrcb[top : top + height, left : left + width])
        pixels = np.rint(rgb * 255.0).astype(np.uint8)
        panel = Image.fromarray(pixels, mode="RGB").resize(
            (width * scale, height * scale), Image.Resampling.NEAREST
        )
        canvas = Image.new("RGB", (panel.width, panel.height + label_height), (24, 24, 24))
        canvas.paste(panel, (0, label_height))
        ImageDraw.Draw(canvas).text((5, 5), label, fill=(255, 255, 255))
        rendered.append(canvas)
    sheet = Image.new(
        "RGB",
        (sum(panel.width for panel in rendered), max(panel.height for panel in rendered)),
        (255, 255, 255),
    )
    offset = 0
    for panel in rendered:
        sheet.paste(panel, (offset, 0))
        offset += panel.width
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return {"top": top, "left": left, "height": height, "width": width}


def _git_revision(project_root: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _git_is_dirty(project_root: Path) -> bool | None:
    process = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(process.stdout.strip()) if process.returncode == 0 else None


def _profile_input_shape(
    learned: LearnedMethod, height: int, width: int
) -> tuple[int, int, int, int]:
    if learned.metadata.get("family") == "v5":
        height += (-height) % 8
        width += (-width) % 8
    return (1, 3, height, width)


def run_benchmark(config: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    manifest_path = _resolve_path(str(config["manifest"]), project_root)
    dataset_root = _resolve_path(str(config["dataset_root"]), project_root)
    output_dir = _resolve_path(str(config["output_dir"]), project_root)
    git_revision_at_start = _git_revision(project_root)
    git_dirty_at_start = _git_is_dirty(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(manifest_path)
    verification = verify_manifest(records, dataset_root)
    if not verification["ok"]:
        raise RuntimeError(f"Manifest verification failed: {verification}")
    test_records = records_for_split(records, str(config.get("split", "test")))
    limit = int(config.get("limit", 0))
    if limit:
        test_records = test_records[:limit]
    device = resolve_device(str(config.get("device", "auto")))
    learned_methods = load_learned_methods(
        list(config.get("learned_methods", [])), project_root, device
    )
    classical_methods = list(config.get("classical_methods", CLASSICAL_METHODS))
    invalid_classical = set(classical_methods) - set(CLASSICAL_METHODS)
    if invalid_classical:
        raise ValueError(f"Unknown classical methods: {sorted(invalid_classical)}")
    degradations = [DegradationSpec(**entry) for entry in config["synthetic_degradations"]]
    for degradation in degradations:
        degradation.validate()
    warmup = int(config.get("profile_warmup", 1))
    repeats = int(config.get("profile_repeats", 3))
    ffmpeg_executable = str(config.get("ffmpeg_executable", "ffmpeg"))
    crop_size = int(config.get("crop_size", 256))
    seed = int(config.get("seed", 2026))
    per_image: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    qualitative_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    model_profiles: dict[tuple[str, int, int], dict[str, int]] = {}

    for index, manifest_record in enumerate(test_records, start=1):
        source_path = dataset_root / manifest_record.relative_path
        try:
            source_rgb, crop = deterministic_crop(
                read_rgb(source_path), manifest_record.id, crop_size, seed
            )
        except ValueError as error:
            skipped.append({"image_id": manifest_record.id, "reason": str(error)})
            continue
        target = rgb_to_ycrcb(source_rgb)
        print(f"[{index}/{len(test_records)}] {manifest_record.id}")
        for degradation in degradations:
            model_input, low_chroma = simulate_420(target, degradation)
            candidates: dict[str, np.ndarray] = {}
            for method in classical_methods:
                function = lambda selected=method: np.concatenate(
                    (
                        target[..., 0:1],
                        classical_reconstruction(
                            selected,
                            target[..., 0],
                            low_chroma,
                            degradation.siting,
                        ),
                    ),
                    axis=2,
                )
                candidate, profile = profile_callable(function, device, warmup, repeats)
                candidate = np.clip(candidate, 0.0, 1.0)
                candidates[method] = candidate
                per_image.append(
                    {
                        "image_id": manifest_record.id,
                        "sha256": manifest_record.sha256,
                        "relative_path": manifest_record.relative_path,
                        "crop": crop,
                        "degradation_id": degradation.name,
                        "degradation": asdict(degradation),
                        "method": method,
                        "metrics": reconstruction_metrics(target, candidate),
                        "profile": profile,
                        "model": None,
                    }
                )
            for learned in learned_methods:
                function = lambda selected=learned: selected.predictor(model_input)
                candidate, profile = profile_callable(function, device, warmup, repeats)
                candidate = np.clip(candidate, 0.0, 1.0)
                candidates[learned.name] = candidate
                profile_key = (learned.name, target.shape[0], target.shape[1])
                if profile_key not in model_profiles:
                    model_profiles[profile_key] = {
                        "parameters": parameter_count(learned.model),
                        "parameter_bytes": parameter_bytes(learned.model),
                        "flops": convolution_flops(
                            learned.model,
                            _profile_input_shape(
                                learned, target.shape[0], target.shape[1]
                            ),
                            device,
                        ),
                    }
                per_image.append(
                    {
                        "image_id": manifest_record.id,
                        "sha256": manifest_record.sha256,
                        "relative_path": manifest_record.relative_path,
                        "crop": crop,
                        "degradation_id": degradation.name,
                        "degradation": asdict(degradation),
                        "method": learned.name,
                        "metrics": reconstruction_metrics(target, candidate),
                        "profile": profile,
                        "model": {
                            **learned.metadata,
                            **model_profiles[profile_key],
                        },
                    }
                )
            baseline_name = "bilinear" if "bilinear" in candidates else classical_methods[0]
            baseline_difficulty = next(
                record["metrics"]["chroma_edge_mae"]
                for record in reversed(per_image)
                if record["image_id"] == manifest_record.id
                and record["degradation_id"] == degradation.name
                and record["method"] == baseline_name
            )
            qualitative_candidates[degradation.name].append(
                {
                    "difficulty": baseline_difficulty,
                    "image_id": manifest_record.id,
                    "target": target.copy(),
                    "candidates": {name: value.copy() for name, value in candidates.items()},
                }
            )

        for quality in config.get("jpeg_qualities", []):
            start = time.perf_counter_ns()
            codec_result = jpeg_roundtrip(source_rgb, int(quality))
            codec_ms = (time.perf_counter_ns() - start) / 1_000_000.0
            codec_input = rgb_to_ycrcb(codec_result.rgb)
            degradation_id = f"jpeg420_q{quality}"
            native_metrics = reconstruction_metrics(target, codec_input)
            per_image.append(
                {
                    "image_id": manifest_record.id,
                    "sha256": manifest_record.sha256,
                    "relative_path": manifest_record.relative_path,
                    "crop": crop,
                    "degradation_id": degradation_id,
                    "degradation": {
                        "codec": codec_result.codec,
                        "quality": int(quality),
                        "encoded_bytes": codec_result.encoded_bytes,
                    },
                    "method": "codec_native",
                    "metrics": native_metrics,
                    "profile": {
                        "runtime_ms_mean": codec_ms,
                        "runtime_ms_median": codec_ms,
                        "runtime_ms_p95": codec_ms,
                        "python_peak_bytes": 0.0,
                        "rss_delta_bytes": 0.0,
                        "accelerator_peak_bytes": 0.0,
                    },
                    "model": None,
                }
            )
            codec_candidates = {"codec_native": codec_input}
            for learned in learned_methods:
                function = lambda selected=learned: selected.predictor(codec_input)
                candidate, profile = profile_callable(function, device, warmup, repeats)
                candidate = np.clip(candidate, 0.0, 1.0)
                codec_candidates[learned.name] = candidate
                profile_key = (learned.name, target.shape[0], target.shape[1])
                per_image.append(
                    {
                        "image_id": manifest_record.id,
                        "sha256": manifest_record.sha256,
                        "relative_path": manifest_record.relative_path,
                        "crop": crop,
                        "degradation_id": degradation_id,
                        "degradation": {
                            "codec": codec_result.codec,
                            "quality": int(quality),
                            "encoded_bytes": codec_result.encoded_bytes,
                        },
                        "method": learned.name,
                        "metrics": reconstruction_metrics(target, candidate),
                        "profile": profile,
                        "model": {
                            **learned.metadata,
                            **model_profiles[profile_key],
                        },
                    }
                )
            qualitative_candidates[degradation_id].append(
                {
                    "difficulty": native_metrics["chroma_edge_mae"],
                    "image_id": manifest_record.id,
                    "target": target.copy(),
                    "candidates": codec_candidates,
                }
            )

        for codec_configuration in config.get("video_codecs", []):
            codec = str(codec_configuration["codec"])
            for quality in codec_configuration["qualities"]:
                degradation_id = f"{codec}_crf{quality}"
                try:
                    start = time.perf_counter_ns()
                    codec_result = video_roundtrip(
                        source_rgb, codec, int(quality), ffmpeg_executable
                    )
                    codec_ms = (time.perf_counter_ns() - start) / 1_000_000.0
                except (RuntimeError, subprocess.CalledProcessError) as error:
                    skipped.append(
                        {
                            "image_id": manifest_record.id,
                            "degradation_id": degradation_id,
                            "reason": str(error),
                        }
                    )
                    continue
                codec_input = rgb_to_ycrcb(codec_result.rgb)
                native_metrics = reconstruction_metrics(target, codec_input)
                per_image.append(
                    {
                        "image_id": manifest_record.id,
                        "sha256": manifest_record.sha256,
                        "relative_path": manifest_record.relative_path,
                        "crop": crop,
                        "degradation_id": degradation_id,
                        "degradation": {
                            "codec": codec,
                            "crf": int(quality),
                            "encoded_bytes": codec_result.encoded_bytes,
                            "command": codec_result.command,
                        },
                        "method": "codec_native",
                        "metrics": native_metrics,
                        "profile": {
                            "runtime_ms_mean": codec_ms,
                            "runtime_ms_median": codec_ms,
                            "runtime_ms_p95": codec_ms,
                            "python_peak_bytes": 0.0,
                            "rss_delta_bytes": 0.0,
                            "accelerator_peak_bytes": 0.0,
                        },
                        "model": None,
                    }
                )
                for learned in learned_methods:
                    candidate, profile = profile_callable(
                        lambda selected=learned: selected.predictor(codec_input),
                        device,
                        warmup,
                        repeats,
                    )
                    candidate = np.clip(candidate, 0.0, 1.0)
                    profile_key = (learned.name, target.shape[0], target.shape[1])
                    per_image.append(
                        {
                            "image_id": manifest_record.id,
                            "sha256": manifest_record.sha256,
                            "relative_path": manifest_record.relative_path,
                            "crop": crop,
                            "degradation_id": degradation_id,
                            "degradation": {
                                "codec": codec,
                                "crf": int(quality),
                                "encoded_bytes": codec_result.encoded_bytes,
                                "command": codec_result.command,
                            },
                            "method": learned.name,
                            "metrics": reconstruction_metrics(target, candidate),
                            "profile": profile,
                            "model": {
                                **learned.metadata,
                                **model_profiles[profile_key],
                            },
                        }
                    )

    if not per_image:
        raise RuntimeError("Benchmark produced no records")
    per_image_path = output_dir / "per_image.jsonl"
    per_image_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in per_image),
        encoding="utf-8",
    )
    aggregate, paired = aggregate_records(
        per_image, int(config.get("bootstrap_samples", 2000)), seed
    )
    qualitative_index = []
    qualitative_count = int(config.get("qualitative_count", 6))
    qualitative_methods = list(config.get("qualitative_methods", []))
    for degradation, entries in qualitative_candidates.items():
        for rank, entry in enumerate(
            sorted(entries, key=lambda item: item["difficulty"], reverse=True)[
                :qualitative_count
            ],
            start=1,
        ):
            center = _gradient_peak(entry["target"])
            selected = entry["candidates"]
            if qualitative_methods:
                selected = {
                    name: value
                    for name, value in selected.items()
                    if name in qualitative_methods
                }
            filename = f"{degradation}_{rank:02d}_{entry['image_id'].replace('/', '_')}.png"
            bounds = save_qualitative_sheet(
                output_dir / "qualitative" / filename,
                entry["target"],
                selected,
                center,
                int(config.get("qualitative_crop_size", 64)),
            )
            qualitative_index.append(
                {
                    "degradation_id": degradation,
                    "rank": rank,
                    "image_id": entry["image_id"],
                    "difficulty": entry["difficulty"],
                    "boundary_center": {"row": center[0], "column": center[1]},
                    "crop": bounds,
                    "file": f"qualitative/{filename}",
                }
            )
    qualitative_directory = output_dir / "qualitative"
    qualitative_directory.mkdir(parents=True, exist_ok=True)
    (qualitative_directory / "index.json").write_text(
        json.dumps(qualitative_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ffmpeg_provenance = dict(ffmpeg_capabilities(ffmpeg_executable))
    if ffmpeg_provenance.get("path"):
        ffmpeg_provenance["path"] = Path(str(ffmpeg_provenance["path"])).name
    report = {
        "protocol": "lossless-disjoint-chroma-benchmark-v1",
        "config": dict(config),
        "provenance": {
            "git_revision": git_revision_at_start,
            "git_dirty_at_start": git_dirty_at_start,
            "manifest": _portable_project_path(manifest_path, project_root),
            "manifest_sha256": sha256_file(manifest_path),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pillow": pillow_version,
            "psutil": psutil.__version__,
            "device": str(device),
            "ffmpeg": ffmpeg_provenance,
        },
        "manifest_verification": verification,
        "evaluated_source_images": len(
            {record["image_id"] for record in per_image}
        ),
        "per_image_records": len(per_image),
        "model_profiles": [
            {
                "method": key[0],
                "height": key[1],
                "width": key[2],
                **profile,
            }
            for key, profile in sorted(model_profiles.items())
        ],
        "skipped": skipped,
        "aggregate": aggregate,
        "paired": paired,
        "qualitative_index": qualitative_index,
        "metric_direction": {
            metric: "higher" if metric in HIGHER_IS_BETTER else "lower"
            for metric in QUALITY_METRICS
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "config.snapshot.json").write_text(
        json.dumps(dict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def run_from_config(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    return run_benchmark(config, Path(project_root).resolve())
