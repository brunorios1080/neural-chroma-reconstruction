"""Optional confidence-filtered EMA teacher/student stage for V7."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .research_benchmark import resolve_device
from .research_data import (
    DegradationSpec,
    ManifestRecord,
    load_manifest,
    read_rgb,
    records_for_split,
    rgb_to_ycrcb,
    simulate_420,
    verify_manifest,
)
from .v7 import load_v7_checkpoint, save_v7_checkpoint
from .v7_losses import V7LossConfig, degrade_chroma_torch, v7_supervised_loss
from .v7_training import V7ManifestCropDataset, _git_revision, _seed_everything


@dataclass(frozen=True)
class SelfTrainingConfig:
    enabled: bool = False
    epochs: int = 5
    batch_size: int = 8
    workers: int = 0
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    ema_alpha: float = 0.995
    minimum_supervised_ratio: float = 0.5
    maximum_pseudo_loss_weight: float = 0.25
    amplitude_scale_max: float = 0.03
    phase_kappa_min: float = 3.0
    forward_error_max: float = 0.02
    augmentation_error_max: float = 0.02
    minimum_acceptance_confidence: float = 0.25
    require_augmentation_consistency: bool = True
    gradient_clip_norm: float = 1.0
    seed: int = 2027
    device: str = "auto"

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.workers < 0:
            raise ValueError("Self-training epochs/batch size must be positive")
        if not 0.0 < self.ema_alpha < 1.0:
            raise ValueError("ema_alpha must be in (0,1)")
        if not 0.0 < self.minimum_supervised_ratio <= 1.0:
            raise ValueError("minimum_supervised_ratio must be in (0,1]")
        if not 0.0 <= self.maximum_pseudo_loss_weight <= 1.0:
            raise ValueError("maximum_pseudo_loss_weight must be in [0,1]")
        for name in (
            "amplitude_scale_max",
            "phase_kappa_min",
            "forward_error_max",
            "augmentation_error_max",
            "minimum_acceptance_confidence",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.minimum_acceptance_confidence > 1.0:
            raise ValueError("minimum_acceptance_confidence cannot exceed one")


class UnlabeledObservationDataset(Dataset):
    """Construct model observations without exposing full-resolution chroma labels."""

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        dataset_root: str | Path,
        crop_size: int,
        degradations: Sequence[DegradationSpec],
        seed: int,
    ) -> None:
        self.records = list(records)
        self.dataset_root = Path(dataset_root)
        self.crop_size = int(crop_size)
        self.degradations = list(degradations)
        self.seed = int(seed)
        self.epoch = 0
        if crop_size < 8 or crop_size % 2:
            raise ValueError("Unlabeled crop_size must be even and >=8")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import hashlib

        record = self.records[index]
        rgb = read_rgb(self.dataset_root / record.relative_path)
        height, width = rgb.shape[:2]
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(f"{record.id} is smaller than the configured crop")
        digest = hashlib.sha256(
            f"{self.seed}:{self.epoch}:{record.id}".encode()
        ).digest()
        generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        top = int(generator.integers(0, height - self.crop_size + 1))
        left = int(generator.integers(0, width - self.crop_size + 1))
        spec_index = int(generator.integers(0, len(self.degradations)))
        # The source is used only to form Y and the simulated low-resolution
        # observation. Its full-resolution chroma is intentionally not returned.
        observed_source = rgb_to_ycrcb(
            rgb[top : top + self.crop_size, left : left + self.crop_size]
        )
        model_input, low = simulate_420(observed_source, self.degradations[spec_index])
        to_chw = lambda value: torch.from_numpy(value).permute(2, 0, 1)
        return to_chw(model_input), to_chw(low), spec_index, record.id


@torch.no_grad()
def _ema_update(
    teacher: torch.nn.Module, student: torch.nn.Module, alpha: float
) -> None:
    teacher_parameters = dict(teacher.named_parameters())
    for name, student_parameter in student.named_parameters():
        teacher_parameters[name].mul_(alpha).add_(student_parameter, alpha=1.0 - alpha)
    teacher_buffers = dict(teacher.named_buffers())
    for name, student_buffer in student.named_buffers():
        teacher_buffers[name].copy_(student_buffer)


@torch.no_grad()
def make_pseudo_labels(
    teacher,
    inputs: torch.Tensor,
    observed_low: torch.Tensor,
    specs: Sequence[DegradationSpec],
    config: SelfTrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Create accepted pseudo targets from uncertainty and consistency criteria."""
    teacher.eval()
    prediction = teacher(inputs)
    forward_maps = []
    for index, spec in enumerate(specs):
        degraded = degrade_chroma_torch(prediction.chroma_mean[index : index + 1], spec)
        low_error = torch.mean(
            torch.abs(degraded - observed_low[index : index + 1]), dim=1, keepdim=True
        )
        forward_maps.append(
            F.interpolate(low_error, size=inputs.shape[-2:], mode="nearest")
        )
    forward_error = torch.cat(forward_maps, dim=0)
    augmentation_error = torch.zeros_like(forward_error)
    if config.require_augmentation_consistency:
        if any(spec.siting != "center" for spec in specs):
            raise ValueError(
                "Horizontal-flip consistency is only valid for center-sited observations"
            )
        flipped = torch.flip(inputs, dims=(-1,))
        flipped_prediction = teacher(flipped).chroma_mean
        aligned = torch.flip(flipped_prediction, dims=(-1,))
        augmentation_error = torch.mean(
            torch.abs(prediction.chroma_mean - aligned), dim=1, keepdim=True
        )
    phase_relevance = prediction.amplitude_mean / (
        prediction.amplitude_mean + teacher.config.safe_phase_amplitude_reference
    )
    combined_confidence = prediction.amplitude_confidence * (
        (1.0 - phase_relevance) + phase_relevance * prediction.phase_confidence
    )
    accepted = (
        (prediction.amplitude_scale <= config.amplitude_scale_max)
        & (prediction.phase_kappa >= config.phase_kappa_min)
        & (forward_error <= config.forward_error_max)
        & (combined_confidence >= config.minimum_acceptance_confidence)
    )
    if config.require_augmentation_consistency:
        accepted &= augmentation_error <= config.augmentation_error_max
    weights = combined_confidence * accepted.to(combined_confidence.dtype)
    stats = {
        "accepted_pixels": float(accepted.sum()),
        "total_pixels": float(accepted.numel()),
        "acceptance_rate": float(accepted.float().mean()),
        "accepted_confidence_mean": (
            float(combined_confidence[accepted].mean()) if accepted.any() else 0.0
        ),
        "forward_error_mean": float(forward_error.mean()),
        "augmentation_disagreement_mean": float(augmentation_error.mean()),
    }
    return prediction.chroma_mean.detach(), weights.detach(), stats


def _assert_unlabeled_disjoint(
    supervised: Sequence[ManifestRecord], unlabeled: Sequence[ManifestRecord]
) -> None:
    scientific_test = [record for record in supervised if record.split == "test"]
    test_ids = {record.id for record in scientific_test}
    test_hashes = {record.sha256 for record in scientific_test}
    overlaps = [
        record.id
        for record in unlabeled
        if record.id in test_ids or record.sha256 in test_hashes
    ]
    if overlaps:
        raise ValueError(
            "Held-out scientific test data cannot be used for pseudo-labels: "
            + ", ".join(overlaps[:5])
        )


def run_v7_self_training(
    configuration: Mapping[str, Any], project_root: str | Path
) -> dict[str, Any]:
    """Run the explicitly enabled second-stage experiment in a separate directory."""
    root = Path(project_root).resolve()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else root / value
    stage = SelfTrainingConfig(**dict(configuration.get("self_training", {})))
    stage.validate()
    if not stage.enabled:
        raise RuntimeError(
            "V7 self-training is disabled; set self_training.enabled=true"
        )
    _seed_everything(stage.seed)
    supervised_manifest = load_manifest(
        resolve(str(configuration["supervised_manifest"]))
    )
    unlabeled_manifest = load_manifest(
        resolve(str(configuration["unlabeled_manifest"]))
    )
    supervised_root = resolve(str(configuration["supervised_dataset_root"]))
    unlabeled_root = resolve(str(configuration["unlabeled_dataset_root"]))
    for records, dataset_root in (
        (supervised_manifest, supervised_root),
        (unlabeled_manifest, unlabeled_root),
    ):
        verification = verify_manifest(records, dataset_root)
        if not verification["ok"]:
            raise RuntimeError(f"Manifest verification failed: {verification}")
    unlabeled_split = str(configuration.get("unlabeled_split", "unlabeled"))
    unlabeled_records = records_for_split(unlabeled_manifest, unlabeled_split)
    _assert_unlabeled_disjoint(supervised_manifest, unlabeled_records)
    supervised_records = records_for_split(supervised_manifest, "train")
    degradations = [DegradationSpec(**entry) for entry in configuration["degradations"]]
    for spec in degradations:
        spec.validate()
    if stage.require_augmentation_consistency and any(
        spec.siting != "center" for spec in degradations
    ):
        raise ValueError(
            "The configured flip-consistency safeguard requires center siting only"
        )
    crop_size = int(configuration.get("crop_size", 256))
    supervised_dataset = V7ManifestCropDataset(
        supervised_records,
        supervised_root,
        crop_size,
        degradations,
        stage.seed,
        True,
    )
    unlabeled_dataset = UnlabeledObservationDataset(
        unlabeled_records,
        unlabeled_root,
        crop_size,
        degradations,
        stage.seed,
    )
    supervised_batch = max(
        1,
        math.ceil(stage.batch_size * stage.minimum_supervised_ratio),
    )
    unlabeled_batch = max(1, stage.batch_size - supervised_batch)
    actual_ratio = supervised_batch / (supervised_batch + unlabeled_batch)
    if actual_ratio < stage.minimum_supervised_ratio:
        raise ValueError("Unable to satisfy minimum_supervised_ratio with batch_size")
    supervised_loader = DataLoader(
        supervised_dataset,
        batch_size=supervised_batch,
        shuffle=True,
        num_workers=stage.workers,
        generator=torch.Generator().manual_seed(stage.seed),
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=unlabeled_batch,
        shuffle=True,
        num_workers=stage.workers,
        generator=torch.Generator().manual_seed(stage.seed + 1),
    )
    device = resolve_device(stage.device)
    checkpoint_path = resolve(str(configuration["teacher_checkpoint"]))
    student, checkpoint = load_v7_checkpoint(checkpoint_path, device)
    teacher, _ = load_v7_checkpoint(checkpoint_path, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    loss_config = V7LossConfig(**dict(checkpoint["loss"]))
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=stage.learning_rate, weight_decay=stage.weight_decay
    )
    output_dir = resolve(str(configuration["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_teacher = output_dir / "teacher_initial.pth"
    if initial_teacher.exists():
        raise FileExistsError(
            f"Refusing to overwrite retained initial teacher: {initial_teacher}"
        )
    shutil.copy2(checkpoint_path, initial_teacher)
    history_path = output_dir / "self_training_history.jsonl"
    history_path.write_text("", encoding="utf-8")
    history = []
    for epoch in range(1, stage.epochs + 1):
        supervised_dataset.set_epoch(epoch)
        unlabeled_dataset.set_epoch(epoch)
        student.train()
        totals = {
            "supervised_loss": 0.0,
            "pseudo_loss": 0.0,
            "accepted_pixels": 0.0,
            "total_pixels": 0.0,
            "confidence": 0.0,
            "teacher_student_disagreement": 0.0,
            "steps": 0.0,
        }
        supervised_iterator = iter(supervised_loader)
        for unlabeled_inputs, unlabeled_low, unlabeled_indices, _ in unlabeled_loader:
            try:
                supervised_batch_data = next(supervised_iterator)
            except StopIteration:
                supervised_iterator = iter(supervised_loader)
                supervised_batch_data = next(supervised_iterator)
            (
                supervised_inputs,
                supervised_targets,
                supervised_low,
                supervised_indices,
                _,
            ) = supervised_batch_data
            supervised_inputs = supervised_inputs.to(device)
            supervised_targets = supervised_targets.to(device)
            supervised_low = supervised_low.to(device)
            unlabeled_inputs = unlabeled_inputs.to(device)
            unlabeled_low = unlabeled_low.to(device)
            supervised_specs = [degradations[int(i)] for i in supervised_indices]
            unlabeled_specs = [degradations[int(i)] for i in unlabeled_indices]
            pseudo_chroma, pseudo_weights, pseudo_stats = make_pseudo_labels(
                teacher, unlabeled_inputs, unlabeled_low, unlabeled_specs, stage
            )
            supervised_loss, _ = v7_supervised_loss(
                student(supervised_inputs),
                supervised_targets,
                student.config.neutral_chroma,
                loss_config,
                supervised_low,
                supervised_specs,
            )
            student_pseudo = student(unlabeled_inputs).chroma_mean
            disagreement = torch.mean(
                torch.abs(student_pseudo - pseudo_chroma), dim=1, keepdim=True
            )
            denominator = pseudo_weights.sum().clamp_min(1e-8)
            pseudo_loss = (pseudo_weights * disagreement).sum() / denominator
            if pseudo_stats["accepted_pixels"] == 0.0:
                pseudo_loss = pseudo_loss * 0.0
            total_loss = (
                supervised_loss + stage.maximum_pseudo_loss_weight * pseudo_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if stage.gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    student.parameters(), stage.gradient_clip_norm
                )
            optimizer.step()
            _ema_update(teacher, student, stage.ema_alpha)
            totals["supervised_loss"] += float(supervised_loss.detach())
            totals["pseudo_loss"] += float(pseudo_loss.detach())
            totals["accepted_pixels"] += pseudo_stats["accepted_pixels"]
            totals["total_pixels"] += pseudo_stats["total_pixels"]
            totals["confidence"] += pseudo_stats["accepted_confidence_mean"]
            totals["teacher_student_disagreement"] += float(
                disagreement.mean().detach()
            )
            totals["steps"] += 1.0
        steps = max(1.0, totals["steps"])
        record = {
            "epoch": epoch,
            "supervised_loss": totals["supervised_loss"] / steps,
            "pseudo_loss": totals["pseudo_loss"] / steps,
            "pseudo_label_acceptance_rate": totals["accepted_pixels"]
            / max(1.0, totals["total_pixels"]),
            "accepted_confidence_mean": totals["confidence"] / steps,
            "teacher_student_disagreement": totals["teacher_student_disagreement"]
            / steps,
            "supervised_sample_ratio": actual_ratio,
            "pseudo_loss_weight": stage.maximum_pseudo_loss_weight,
        }
        history.append(record)
        with history_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        extra = {
            "stage": "self_training",
            "self_training": asdict(stage),
            "teacher_model": teacher.state_dict(),
            "source_supervised_checkpoint": str(configuration["teacher_checkpoint"]),
            "initial_teacher_checkpoint": "teacher_initial.pth",
            "degradations": [asdict(spec) for spec in degradations],
        }
        save_v7_checkpoint(
            output_dir / "student_last.pth",
            student,
            epoch,
            stage.seed,
            asdict(loss_config),
            optimizer,
            _git_revision(root),
            extra,
        )
        print(
            f"V7 self-train {epoch}/{stage.epochs}: supervised={record['supervised_loss']:.6f} "
            f"pseudo={record['pseudo_loss']:.6f} accepted={record['pseudo_label_acceptance_rate']:.3f}"
        )
    return {"output_dir": str(output_dir), "history": history}
