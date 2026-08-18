"""Deterministic training loop for the V6 ablation matrix."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
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
from .research_models import (
    AblationConfig,
    build_ablation_model,
    chroma_loss,
    save_ablation_checkpoint,
)


def _seed_from_text(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


class ManifestCropDataset(Dataset):
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
        self.crop_size = crop_size
        self.degradations = list(degradations)
        self.seed = seed
        self.random_crop = random_crop
        self.epoch = 0
        if crop_size < 8 or crop_size % 2:
            raise ValueError("Training crop_size must be an even integer of at least 8")
        if not self.degradations:
            raise ValueError("At least one training degradation is required")

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
                f"{record.id} is {width}x{height}, smaller than {self.crop_size}"
            )
        generator = np.random.default_rng(
            _seed_from_text(f"{self.seed}:{self.epoch}:{record.id}")
        )
        if self.random_crop:
            top = int(generator.integers(0, height - self.crop_size + 1))
            left = int(generator.integers(0, width - self.crop_size + 1))
        else:
            top = (height - self.crop_size) // 2
            left = (width - self.crop_size) // 2
        crop = rgb[top : top + self.crop_size, left : left + self.crop_size]
        target = rgb_to_ycrcb(crop)
        degradation = self.degradations[
            int(generator.integers(0, len(self.degradations)))
            if self.random_crop
            else index % len(self.degradations)
        ]
        model_input, _ = simulate_420(target, degradation)
        input_tensor = torch.from_numpy(model_input).permute(2, 0, 1)
        target_tensor = torch.from_numpy(target).permute(2, 0, 1)
        return input_tensor, target_tensor, record.id, degradation.name


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: AblationConfig,
) -> float:
    model.eval()
    total = 0.0
    samples = 0
    for inputs, targets, _, _ in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        loss = chroma_loss(model(inputs), targets, config)
        total += float(loss.item()) * inputs.shape[0]
        samples += inputs.shape[0]
    if samples == 0:
        raise RuntimeError("Validation split produced no samples")
    return total / samples


def train_ablation(
    config: AblationConfig,
    manifest_records: Sequence[ManifestRecord],
    dataset_root: str | Path,
    output_dir: str | Path,
    degradations: Sequence[DegradationSpec],
    training: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    config.validate()
    seed = int(training.get("seed", 2026))
    _seed_everything(seed)
    train_records = records_for_split(manifest_records, "train")
    validation_split = str(training.get("validation_split", "validation"))
    validation_records = records_for_split(manifest_records, validation_split)
    crop_size = int(training.get("crop_size", 256))
    train_dataset = ManifestCropDataset(
        train_records, dataset_root, crop_size, degradations, seed, True
    )
    validation_dataset = ManifestCropDataset(
        validation_records, dataset_root, crop_size, degradations, seed, False
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(training.get("workers", 0)),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(training.get("workers", 0)),
    )
    model = build_ablation_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = int(training.get("epochs", 30))
    destination = Path(output_dir) / config.name
    destination.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        running = 0.0
        samples = 0
        for inputs, targets, _, _ in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = chroma_loss(model(inputs), targets, config)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * inputs.shape[0]
            samples += inputs.shape[0]
        training_loss = running / max(1, samples)
        validation_loss = validate(model, validation_loader, device, config)
        record = {
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
        }
        history.append(record)
        print(
            f"{config.name} epoch {epoch}/{epochs}: "
            f"train={training_loss:.6f} validation={validation_loss:.6f}"
        )
        checkpoint_extra = {
            "best_validation_loss": min(best_loss, validation_loss),
            "training": dict(training),
            "degradations": [asdict(spec) for spec in degradations],
        }
        save_ablation_checkpoint(
            destination / "last.pth",
            model,
            epoch,
            optimizer,
            checkpoint_extra,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_ablation_checkpoint(
                destination / "best.pth",
                model,
                epoch,
                optimizer,
                checkpoint_extra,
            )
    (destination / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "name": config.name,
        "config": asdict(config),
        "best_validation_loss": best_loss,
        "best_checkpoint": str(destination / "best.pth"),
        "last_checkpoint": str(destination / "last.pth"),
        "history": history,
    }


def run_ablation_matrix(
    matrix: Mapping[str, Any],
    manifest_records: Sequence[ManifestRecord],
    dataset_root: str | Path,
    output_dir: str | Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    verification = verify_manifest(manifest_records, dataset_root)
    if not verification["ok"]:
        raise RuntimeError(f"Manifest verification failed: {verification}")
    degradations = [DegradationSpec(**entry) for entry in matrix["degradations"]]
    experiments = [AblationConfig(**entry) for entry in matrix["experiments"]]
    results = []
    for experiment in experiments:
        results.append(
            train_ablation(
                experiment,
                manifest_records,
                dataset_root,
                output_dir,
                degradations,
                matrix["training"],
                device,
            )
        )
    working_directory = Path.cwd().resolve()
    for result in results:
        for field in ("best_checkpoint", "last_checkpoint"):
            checkpoint = Path(result[field]).resolve()
            try:
                result[field] = checkpoint.relative_to(working_directory).as_posix()
            except ValueError:
                result[field] = str(checkpoint)
    destination = Path(output_dir)
    (destination / "matrix_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    learned_methods = [
        {
            "name": result["name"],
            "type": "ablation",
            "weights": result["best_checkpoint"],
        }
        for result in results
    ]
    (destination / "learned_methods.json").write_text(
        json.dumps(learned_methods, indent=2) + "\n", encoding="utf-8"
    )
    return results
