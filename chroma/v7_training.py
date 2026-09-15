"""Manifest-driven supervised training implementation for V7."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .research_data import (
    DegradationSpec,
    ManifestRecord,
    read_rgb,
    records_for_split,
    rgb_to_ycrcb,
    simulate_420,
    verify_manifest,
)
from .training import _autocast, _make_scaler
from .v7 import V7Config, V7PolarChromaRefiner, load_v7_checkpoint, save_v7_checkpoint
from .v7_losses import V7LossConfig, v7_supervised_loss


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_revision(project_root: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


class V7ManifestCropDataset(Dataset):
    """Publication crops retaining the actual low-resolution observation/spec."""

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        dataset_root: str | Path,
        crop_size: int,
        degradations: Sequence[DegradationSpec],
        seed: int,
        random_crop: bool,
    ) -> None:
        self.records = list(records)
        self.dataset_root = Path(dataset_root)
        self.crop_size = int(crop_size)
        self.degradations = list(degradations)
        self.seed = int(seed)
        self.random_crop = bool(random_crop)
        self.epoch = 0
        if self.crop_size < 8 or self.crop_size % 2:
            raise ValueError("V7 crop_size must be an even integer >=8")
        if not self.degradations:
            raise ValueError("V7 requires at least one degradation")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        rgb = read_rgb(self.dataset_root / record.relative_path)
        height, width = rgb.shape[:2]
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(
                f"{record.id} is smaller than {self.crop_size}x{self.crop_size}"
            )
        digest = hashlib.sha256(
            f"{self.seed}:{self.epoch}:{record.id}".encode()
        ).digest()
        generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        if self.random_crop:
            top = int(generator.integers(0, height - self.crop_size + 1))
            left = int(generator.integers(0, width - self.crop_size + 1))
            spec_index = int(generator.integers(0, len(self.degradations)))
        else:
            top = (height - self.crop_size) // 2
            left = (width - self.crop_size) // 2
            spec_index = index % len(self.degradations)
        target = rgb_to_ycrcb(
            rgb[top : top + self.crop_size, left : left + self.crop_size]
        )
        model_input, low = simulate_420(target, self.degradations[spec_index])
        to_chw = lambda value: torch.from_numpy(value).permute(2, 0, 1)
        return (
            to_chw(model_input),
            to_chw(target),
            to_chw(low),
            spec_index,
            record.id,
        )


def _ycrcb_to_rgb_tensor(ycrcb: torch.Tensor) -> torch.Tensor:
    """Convert full-range BT.601 Y, Cr, Cb BCHW tensors to clipped RGB."""
    luma = ycrcb[:, 0:1]
    cr = ycrcb[:, 1:2] - 0.5
    cb = ycrcb[:, 2:3] - 0.5
    return torch.cat(
        (
            luma + 1.403 * cr,
            luma - 0.714 * cr - 0.344 * cb,
            luma + 1.773 * cb,
        ),
        dim=1,
    ).clamp(0.0, 1.0)


def _psnr_ssim(
    reference: torch.Tensor, candidate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-image PSNR and standard Gaussian-window SSIM."""
    if reference.shape != candidate.shape or reference.ndim != 4:
        raise ValueError("PSNR/SSIM inputs must be matching BCHW tensors")
    squared_error = (reference - candidate).square().flatten(1).mean(1)
    psnr = (10.0 * torch.log10(1.0 / squared_error.clamp_min(1e-8))).clamp_max(
        80.0
    )

    height, width = reference.shape[-2:]
    window_size = min(
        11,
        height if height % 2 else height - 1,
        width if width % 2 else width - 1,
    )
    window_size = max(3, window_size)
    sigma = 1.5 * window_size / 11.0
    coordinates = torch.arange(
        window_size, device=reference.device, dtype=reference.dtype
    )
    coordinates = coordinates - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    channels = reference.shape[1]
    kernel = kernel_2d.expand(channels, 1, window_size, window_size)
    padding = window_size // 2

    def blur(value: torch.Tensor) -> torch.Tensor:
        padded = F.pad(value, (padding,) * 4, mode="reflect")
        return F.conv2d(padded, kernel, groups=channels)

    mu_reference = blur(reference)
    mu_candidate = blur(candidate)
    mu_reference_sq = mu_reference.square()
    mu_candidate_sq = mu_candidate.square()
    mu_product = mu_reference * mu_candidate
    variance_reference = blur(reference.square()) - mu_reference_sq
    variance_candidate = blur(candidate.square()) - mu_candidate_sq
    covariance = blur(reference * candidate) - mu_product
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2.0 * mu_product + c1) * (2.0 * covariance + c2)
    denominator = (mu_reference_sq + mu_candidate_sq + c1) * (
        variance_reference + variance_candidate + c2
    )
    ssim = (numerator / denominator.clamp_min(torch.finfo(reference.dtype).eps))
    return psnr, ssim.flatten(1).mean(1)


@torch.no_grad()
def validate_v7(
    model: V7PolarChromaRefiner,
    loader: DataLoader,
    degradations: Sequence[DegradationSpec],
    loss_config: V7LossConfig,
    device: torch.device,
    non_blocking: bool = False,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    samples = 0
    for inputs, targets, low, spec_indices, _ in loader:
        inputs = inputs.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        low = low.to(device, non_blocking=non_blocking)
        specs = [degradations[int(index)] for index in spec_indices]
        prediction = model(inputs)
        _, components = v7_supervised_loss(
            prediction,
            targets,
            model.config.neutral_chroma,
            loss_config,
            low,
            specs,
        )
        count = inputs.shape[0]
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value) * count
        target_rgb = _ycrcb_to_rgb_tensor(targets)
        predicted_ycrcb = prediction.ycrcb("mean")
        predicted_rgb = _ycrcb_to_rgb_tensor(predicted_ycrcb)
        chroma_psnr, chroma_ssim = _psnr_ssim(
            targets[:, 1:3], prediction.chroma_mean
        )
        rgb_psnr, rgb_ssim = _psnr_ssim(target_rgb, predicted_rgb)
        totals["chroma_psnr"] = totals.get("chroma_psnr", 0.0) + float(
            chroma_psnr.sum()
        )
        totals["chroma_ssim"] = totals.get("chroma_ssim", 0.0) + float(
            chroma_ssim.sum()
        )
        totals["rgb_psnr"] = totals.get("rgb_psnr", 0.0) + float(rgb_psnr.sum())
        totals["rgb_ssim"] = totals.get("rgb_ssim", 0.0) + float(rgb_ssim.sum())
        samples += count
    if samples == 0:
        raise RuntimeError("V7 validation produced no samples")
    return {name: value / samples for name, value in totals.items()}


def train_v7(
    configuration: Mapping[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    """Train one V7 experiment; callers decide when to invoke this operation."""
    root = Path(project_root).resolve()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else root / value
    manifest_path = resolve(str(configuration["manifest"]))
    dataset_root = resolve(str(configuration["dataset_root"]))
    output_dir = resolve(str(configuration["output_dir"]))
    from .research_benchmark import resolve_device
    from .research_data import load_manifest

    records = load_manifest(manifest_path)
    training = dict(configuration.get("training", {}))
    if bool(training.get("verify_manifest", True)):
        verification = verify_manifest(records, dataset_root)
        if not verification["ok"]:
            raise RuntimeError(f"Manifest verification failed: {verification}")
    seed = int(training.get("seed", 2026))
    _seed_everything(seed)
    model_config = V7Config(**dict(configuration.get("model", {})))
    loss_config = V7LossConfig(**dict(configuration.get("loss", {})))
    model_config.validate()
    loss_config.validate()
    degradations = [DegradationSpec(**entry) for entry in configuration["degradations"]]
    for spec in degradations:
        spec.validate()
    crop_size = int(training.get("crop_size", 256))
    train_records = records_for_split(
        records, str(training.get("train_split", "train"))
    )
    validation_records = records_for_split(
        records, str(training.get("validation_split", "validation"))
    )
    train_dataset = V7ManifestCropDataset(
        train_records, dataset_root, crop_size, degradations, seed, True
    )
    validation_dataset = V7ManifestCropDataset(
        validation_records, dataset_root, crop_size, degradations, seed, False
    )
    device = resolve_device(str(training.get("device", "auto")))
    pin_memory = bool(training.get("pin_memory", device.type == "cuda"))
    workers = int(training.get("workers", 0))
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": int(training.get("batch_size", 16)),
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        loader_options["persistent_workers"] = bool(
            training.get("persistent_workers", True)
        )
        loader_options["prefetch_factor"] = int(training.get("prefetch_factor", 2))
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_options
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    amp = bool(training.get("amp", True) and device.type == "cuda")
    if device.type == "cuda":
        allow_tf32 = bool(training.get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = bool(training.get("cudnn_benchmark", True))
    model = V7PolarChromaRefiner(model_config).to(device)
    start_epoch = 1
    best = float("inf")
    resume = training.get("resume")
    if resume:
        restored, checkpoint = load_v7_checkpoint(
            resolve(str(resume)), device, expected_config=model_config
        )
        model.load_state_dict(restored.state_dict(), strict=True)
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best_validation_loss", best))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    if resume and isinstance(checkpoint.get("optimizer"), dict):
        optimizer.load_state_dict(checkpoint["optimizer"])
    scaler = _make_scaler(amp)
    if resume and isinstance(checkpoint.get("scaler"), dict):
        scaler.load_state_dict(checkpoint["scaler"])
    epochs = int(training.get("epochs", 30))
    if epochs < start_epoch:
        raise ValueError("Configured epochs precede the resume checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    if not resume:
        history_path.write_text("", encoding="utf-8")
    history = []
    for epoch in range(start_epoch, epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        totals: dict[str, float] = {}
        samples = 0
        for inputs, targets, low, spec_indices, _ in train_loader:
            inputs = inputs.to(device, non_blocking=pin_memory)
            targets = targets.to(device, non_blocking=pin_memory)
            low = low.to(device, non_blocking=pin_memory)
            specs = [degradations[int(index)] for index in spec_indices]
            with _autocast(device, amp):
                loss, components = v7_supervised_loss(
                    model(inputs),
                    targets,
                    model.config.neutral_chroma,
                    loss_config,
                    low,
                    specs,
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            gradient_clip = float(training.get("gradient_clip_norm", 1.0))
            if gradient_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            count = inputs.shape[0]
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * count
            samples += count
        train_metrics = {
            name: value / max(1, samples) for name, value in totals.items()
        }
        validation_metrics = validate_v7(
            model,
            validation_loader,
            degradations,
            loss_config,
            device,
            non_blocking=pin_memory,
        )
        record = {
            "epoch": epoch,
            "training": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        with history_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        score = validation_metrics["cartesian_l1"]
        extra = {
            "best_validation_loss": min(best, score),
            "training": training,
            "degradations": [asdict(spec) for spec in degradations],
            "manifest": str(configuration["manifest"]),
            "scaler": scaler.state_dict(),
            "epoch_metrics": record,
        }
        save_v7_checkpoint(
            output_dir / "last.pth",
            model,
            epoch,
            seed,
            asdict(loss_config),
            optimizer,
            _git_revision(root),
            extra,
        )
        if score < best:
            best = score
            save_v7_checkpoint(
                output_dir / "best.pth",
                model,
                epoch,
                seed,
                asdict(loss_config),
                optimizer,
                _git_revision(root),
                extra,
            )
        print(
            f"V7 epoch {epoch}/{epochs}: train={train_metrics['total']:.6f} "
            f"validation_cart={score:.6f} "
            f"chroma_psnr={validation_metrics['chroma_psnr']:.2f}dB "
            f"chroma_ssim={validation_metrics['chroma_ssim']:.4f}"
        )
    return {
        "output_dir": str(output_dir),
        "best_validation_loss": best,
        "best_checkpoint": str(output_dir / "best.pth"),
        "last_checkpoint": str(output_dir / "last.pth"),
        "history": history,
    }


def run_v7_ablation_matrix(
    matrix: Mapping[str, Any], project_root: str | Path
) -> list[dict[str, Any]]:
    """Run the V6 control and V7 variants under one matched protocol."""
    shared = {key: value for key, value in matrix.items() if key != "experiments"}
    root = Path(project_root).resolve()
    destination = Path(str(matrix["output_dir"]))
    if not destination.is_absolute():
        destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)
    control_results = []
    controls = list(matrix.get("controls", []))
    if controls:
        from .research_benchmark import resolve_device
        from .research_data import load_manifest
        from .research_models import AblationConfig
        from .research_training import train_ablation

        manifest_path = Path(str(matrix["manifest"]))
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        dataset_root = Path(str(matrix["dataset_root"]))
        if not dataset_root.is_absolute():
            dataset_root = root / dataset_root
        records = load_manifest(manifest_path)
        verification = verify_manifest(records, dataset_root)
        if not verification["ok"]:
            raise RuntimeError(f"Manifest verification failed: {verification}")
        degradations = [DegradationSpec(**entry) for entry in matrix["degradations"]]
        device = resolve_device(str(matrix["training"].get("device", "auto")))
        for control in controls:
            ablation = {"name": str(control["name"]), **dict(control["ablation"])}
            result = train_ablation(
                AblationConfig(**ablation),
                records,
                dataset_root,
                destination,
                degradations,
                matrix["training"],
                device,
            )
            result["output_dir"] = str(destination / str(control["name"]))
            control_results.append(result)
    v7_results = []
    for experiment in matrix["experiments"]:
        config = dict(shared)
        config["model"] = {
            **dict(shared.get("model", {})),
            **dict(experiment.get("model", {})),
        }
        config["loss"] = {
            **dict(shared.get("loss", {})),
            **dict(experiment.get("loss", {})),
        }
        base_output = Path(str(shared["output_dir"]))
        config["output_dir"] = str(base_output / str(experiment["name"]))
        result = train_v7(config, root)
        result["name"] = experiment["name"]
        v7_results.append(result)
    results = [*control_results, *v7_results]

    def portable(path_value: str) -> str:
        path = Path(path_value).resolve()
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return str(path)

    serialized_results = []
    for result in results:
        serialized = dict(result)
        serialized["output_dir"] = portable(result["output_dir"])
        serialized["best_checkpoint"] = portable(result["best_checkpoint"])
        serialized["last_checkpoint"] = portable(result["last_checkpoint"])
        serialized_results.append(serialized)
    (destination / "matrix_results.json").write_text(
        json.dumps(serialized_results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    learned_methods = [
        {
            "name": result["name"],
            "type": "ablation",
            "weights": portable(result["best_checkpoint"]),
        }
        for result in control_results
    ] + [
        {
            "name": result["name"],
            "type": "v7",
            "mode": "mean",
            "weights": portable(result["best_checkpoint"]),
        }
        for result in v7_results
    ]
    probabilistic = next(
        (result for result in v7_results if result["name"] == "v7_probabilistic"),
        None,
    )
    if probabilistic is not None:
        learned_methods.append(
            {
                "name": "v7_probabilistic_safe",
                "type": "v7",
                "mode": "safe",
                "weights": portable(probabilistic["best_checkpoint"]),
            }
        )
    (destination / "learned_methods.json").write_text(
        json.dumps(learned_methods, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results
