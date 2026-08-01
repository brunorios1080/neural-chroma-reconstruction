"""Shared, resumable training pipeline for the V5 and V6 models."""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoints import load_model, save_checkpoint, strip_module_prefix, unwrap
from .data import (
    YUVChromaDataset,
    collate_valid,
    list_image_files,
    seed_worker,
    split_files,
)
from .models import Discriminator, build_model, normalize_version, parameter_count
from .visualization import save_comparison


@dataclass
class TrainingConfig:
    model_version: str
    source: Path
    output_dir: Path
    samples_dir: Path
    epochs: int = 30
    batch_size: int = 16
    crop_size: int = 256
    workers: int = 4
    val_fraction: float = 0.03
    seed: int = 1337
    learning_rate: float | None = None
    device: str = "auto"
    resume: Path | None = None
    amp: bool = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, enabled=True)


def make_loader(
    dataset: YUVChromaDataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        collate_fn=collate_valid,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _move_batch(batch, device: torch.device):
    packed, failures = batch
    if packed is None:
        return None, failures
    inputs, targets, paths = packed
    return (
        inputs.to(device, non_blocking=True),
        targets.to(device, non_blocking=True),
        paths,
    ), failures


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader | None,
    device: torch.device,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    if loader is None:
        return {}, None
    model.eval()
    chroma_total = 0.0
    full_total = 0.0
    sample_count = 0
    example = None
    for raw_batch in loader:
        batch, _ = _move_batch(raw_batch, device)
        if batch is None:
            continue
        inputs, targets, _ = batch
        predictions = model(inputs)
        count = inputs.shape[0]
        chroma_total += (
            F.l1_loss(predictions[:, 1:], targets[:, 1:], reduction="mean").item()
            * count
        )
        full_total += F.l1_loss(predictions, targets, reduction="mean").item() * count
        sample_count += count
        if example is None:
            example = (inputs[0], targets[0], predictions[0])
    if sample_count == 0:
        raise RuntimeError(
            "Validation produced no usable batches; check image sizes and files"
        )
    return {
        "chroma_l1": chroma_total / sample_count,
        "full_l1": full_total / sample_count,
    }, example


def train_v6_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    amp: bool,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total, samples = 0.0, 0
    failures = 0
    progress = tqdm(loader, desc=f"V6 epoch {epoch}")
    for raw_batch in progress:
        batch, failed = _move_batch(raw_batch, device)
        failures += len(failed)
        if batch is None:
            continue
        inputs, targets, _ = batch
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            predictions = model(inputs)
            loss = F.l1_loss(predictions[:, 1:], targets[:, 1:])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        count = inputs.shape[0]
        total += loss.item() * count
        samples += count
        progress.set_postfix(chroma_l1=f"{loss.item():.6f}", skipped=failures)
    if samples == 0:
        raise RuntimeError(
            "Training produced no usable batches; check image sizes and files"
        )
    return {"chroma_l1": total / samples, "skipped": float(failures)}


def train_v5_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    amp: bool,
    epoch: int,
) -> dict[str, float]:
    generator.train()
    discriminator.train()
    generator_total = discriminator_total = 0.0
    chroma_total = 0.0
    samples = failures = 0
    progress = tqdm(loader, desc=f"V5 epoch {epoch}")
    for raw_batch in progress:
        batch, failed = _move_batch(raw_batch, device)
        failures += len(failed)
        if batch is None:
            continue
        inputs, targets, _ = batch

        discriminator_optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            with torch.no_grad():
                detached_predictions = generator(inputs)
            real_logits = discriminator(targets)
            fake_logits = discriminator(detached_predictions)
            discriminator_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    real_logits, torch.ones_like(real_logits)
                )
                + F.binary_cross_entropy_with_logits(
                    fake_logits, torch.zeros_like(fake_logits)
                )
            )
        scaler.scale(discriminator_loss).backward()
        scaler.step(discriminator_optimizer)

        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
        generator_optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            predictions = generator(inputs)
            fake_logits = discriminator(predictions)
            adversarial_loss = F.binary_cross_entropy_with_logits(
                fake_logits, torch.ones_like(fake_logits)
            )
            reconstruction_loss = F.l1_loss(predictions, targets)
            generator_loss = adversarial_loss + 10.0 * reconstruction_loss
        scaler.scale(generator_loss).backward()
        scaler.step(generator_optimizer)
        scaler.update()
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)

        count = inputs.shape[0]
        generator_total += generator_loss.item() * count
        discriminator_total += discriminator_loss.item() * count
        chroma_total += F.l1_loss(predictions[:, 1:], targets[:, 1:]).item() * count
        samples += count
        progress.set_postfix(
            generator=f"{generator_loss.item():.4f}",
            discriminator=f"{discriminator_loss.item():.4f}",
            skipped=failures,
        )
    if samples == 0:
        raise RuntimeError(
            "Training produced no usable batches; check image sizes and files"
        )
    return {
        "generator": generator_total / samples,
        "discriminator": discriminator_total / samples,
        "chroma_l1": chroma_total / samples,
        "skipped": float(failures),
    }


def _restore_training_state(
    checkpoint: dict[str, Any],
    version: str,
    discriminator: nn.Module | None,
    optimizers: dict[str, torch.optim.Optimizer],
) -> tuple[int, float]:
    saved_version = checkpoint.get("model_version")
    if saved_version and normalize_version(str(saved_version)) != version:
        raise ValueError(
            f"Cannot resume {version} training from {saved_version} checkpoint"
        )
    if discriminator is not None:
        state = checkpoint.get("discriminator", checkpoint.get("D"))
        if isinstance(state, dict):
            discriminator.load_state_dict(strip_module_prefix(state))
    saved_optimizers = checkpoint.get("optimizers", {})
    if isinstance(saved_optimizers, dict):
        for name, optimizer in optimizers.items():
            state = saved_optimizers.get(name)
            if isinstance(state, dict):
                optimizer.load_state_dict(state)
    return int(checkpoint.get("epoch", 0)) + 1, float(
        checkpoint.get("best_chroma_l1", "inf")
    )


def run_training(config: TrainingConfig) -> None:
    version = normalize_version(config.model_version)
    if config.epochs < 1 or config.batch_size < 1 or config.workers < 0:
        raise ValueError(
            "epochs and batch_size must be positive; workers cannot be negative"
        )
    if version == "v5" and (config.crop_size < 64 or config.crop_size % 8 != 0):
        raise ValueError("V5 crop size must be at least 64 and divisible by 8")

    set_seed(config.seed)
    device = resolve_device(config.device)
    amp = bool(config.amp and device.type == "cuda")
    files = list_image_files(config.source)
    train_files, val_files = split_files(files, config.val_fraction, config.seed)
    train_dataset = YUVChromaDataset(train_files, config.crop_size, random_crop=True)
    val_dataset = YUVChromaDataset(val_files, config.crop_size, random_crop=False)
    train_loader = make_loader(
        train_dataset,
        config.batch_size,
        True,
        config.workers,
        config.seed,
        device.type == "cuda",
    )
    val_loader = (
        make_loader(
            val_dataset,
            config.batch_size,
            False,
            config.workers,
            config.seed,
            device.type == "cuda",
        )
        if val_files
        else None
    )

    model = build_model(version).to(device)
    discriminator = Discriminator().to(device) if version == "v5" else None
    learning_rate = config.learning_rate or (2e-4 if version == "v5" else 1e-4)
    if version == "v5":
        optimizers = {
            "generator": torch.optim.Adam(
                model.parameters(), lr=learning_rate, betas=(0.5, 0.999)
            ),
            "discriminator": torch.optim.Adam(
                discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
            ),
        }
    else:
        optimizers = {"model": torch.optim.AdamW(model.parameters(), lr=learning_rate)}

    start_epoch, best_chroma_l1 = 1, float("inf")
    if config.resume:
        checkpoint = load_model(model, config.resume, version, device)
        start_epoch, best_chroma_l1 = _restore_training_state(
            checkpoint, version, discriminator, optimizers
        )

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if discriminator is not None:
            discriminator = nn.DataParallel(discriminator)
    scaler = _make_scaler(amp)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.samples_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Model: {version.upper()} | Parameters: {parameter_count(unwrap(model)):,} | "
        f"Device: {device} | Train: {len(train_files)} | Validation: {len(val_files)}"
    )

    history_path = config.output_dir / "history.jsonl"
    if not config.resume:
        history_path.write_text("", encoding="utf-8")
    for epoch in range(start_epoch, config.epochs + 1):
        if version == "v5":
            train_metrics = train_v5_epoch(
                model,
                discriminator,
                train_loader,
                optimizers["generator"],
                optimizers["discriminator"],
                scaler,
                device,
                amp,
                epoch,
            )
        else:
            train_metrics = train_v6_epoch(
                model, train_loader, optimizers["model"], scaler, device, amp, epoch
            )
        val_metrics, example = validate(model, val_loader, device)
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        print(json.dumps(record, sort_keys=True))
        with history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(record, sort_keys=True) + "\n")

        if example is not None:
            save_comparison(*example, config.samples_dir / f"epoch_{epoch:03d}.png")
        current_score = val_metrics.get(
            "chroma_l1", train_metrics.get("chroma_l1", float("inf"))
        )
        payload = {
            "format_version": 1,
            "model_version": version,
            "epoch": epoch,
            "model": unwrap(model).state_dict(),
            "discriminator": unwrap(discriminator).state_dict()
            if discriminator is not None
            else None,
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in optimizers.items()
            },
            "best_chroma_l1": min(best_chroma_l1, current_score),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(config).items()
            },
        }
        save_checkpoint(config.output_dir / "last.pth", payload)
        save_checkpoint(config.output_dir / f"epoch_{epoch:03d}.pth", payload)
        if current_score < best_chroma_l1:
            best_chroma_l1 = current_score
            save_checkpoint(config.output_dir / "best.pth", payload)
