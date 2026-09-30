"""Train the V5 generator on the exact data protocol of a completed Prism run."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .models import UNetGenerator
from .prism_data import EpochSampler, PrismDataset, audit_manifest
from .prism_metrics import quality_batch
from .prism_training import atomic_json, save_checkpoint
from .research_data import DegradationSpec, load_manifest


def chroma_mse(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Per-image chroma MSE on raw YCrCb values, matching Prism's loss domain."""
    return (target[:, 1:3] - prediction[:, 1:3]).square().flatten(1).mean(1)


def preserve_luma(inputs: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Use the observed luminance, as every deterministic Prism model does."""
    return torch.cat((inputs[:, :1], prediction[:, 1:3]), dim=1)


class MatchedV5Inference(nn.Module):
    """Apply the luminance passthrough used while training this V5 generator."""

    def __init__(self, generator: UNetGenerator):
        super().__init__()
        self.generator = generator

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return preserve_luma(inputs, self.generator(inputs))


def load_matched_v5(path, device="cpu"):
    """Load a matched-run checkpoint with its required output transform."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("run_name") != "v5_prism_mse":
        raise ValueError("Expected a V5 Prism MSE checkpoint")
    generator = UNetGenerator()
    generator.load_state_dict(payload["model"], strict=True)
    return MatchedV5Inference(generator).to(device).eval(), payload


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    totals, baseline_totals, groups = defaultdict(float), defaultdict(float), {}
    samples = 0
    for inputs, targets, _, indices, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        prediction = preserve_luma(inputs, model(inputs))
        metrics = quality_batch(targets, prediction)
        baseline = quality_batch(targets, inputs)
        metrics["chroma_mse"] = chroma_mse(targets, prediction)
        baseline["chroma_mse"] = chroma_mse(targets, inputs)
        for name, values in metrics.items():
            if not torch.isfinite(values).all():
                raise FloatingPointError(f"Non-finite validation metric: {name}")
            totals[name] += float(values.sum())
        for name, values in baseline.items():
            baseline_totals[name] += float(values.sum())
        for index in indices.unique().tolist():
            mask = indices == index
            group = groups.setdefault(int(index), {"samples": 0, "chroma_mse": 0.0})
            group["samples"] += int(mask.sum())
            group["chroma_mse"] += float(metrics["chroma_mse"][mask].sum())
        samples += len(inputs)
    if not samples:
        raise ValueError("Validation has no samples")
    return (
        {name: total / samples for name, total in totals.items()},
        {name: total / samples for name, total in baseline_totals.items()},
        groups,
        samples,
    )


def train(reference_run, dataset_root, output_dir, workers=None, resume=False,
          stop_after_epoch=None, device_name="cuda"):
    reference_run = Path(reference_run).resolve()
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == reference_run:
        raise ValueError("V5 output must be separate from the Prism reference run")
    reference = json.loads((reference_run / "config.snapshot.json").read_text())
    if reference["loss"]["reconstruction"] != "mse":
        raise ValueError("Reference Prism run must use MSE reconstruction")
    manifest_bytes = (reference_run / "manifest.jsonl").read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    records = load_manifest(reference_run / "manifest.jsonl")
    settings = reference["training"]
    epochs = int(settings["epochs"])
    batch_size = int(settings["batch_size"])
    crop_size = int(settings["crop_size"])
    seed = int(settings["seed"])
    workers = int(settings["workers"] if workers is None else workers)
    if epochs != 100 or batch_size < 1 or workers < 0 or crop_size % 8:
        raise ValueError("Expected a 100-epoch Prism run with valid batch/workers/V5 crop")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= epochs:
        raise ValueError("stop_after_epoch must be within the run")
    specs = [DegradationSpec(**entry) for entry in reference["degradations"]]
    for spec in specs:
        spec.validate()
    # The reference run hashed the data while preparing its manifest. This run
    # stages a fresh archive, so verify every image against that saved manifest.
    audit_manifest(records, dataset_root, reference["data_kind"], crop_size, True)
    if not any(row.split == "train" for row in records) or not any(
        row.split == "validation" for row in records
    ):
        raise ValueError("Reference manifest requires train and validation records")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        tf32 = bool(settings.get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    contract = {
        "model": "v5_unet_generator",
        "objective": "chroma_mse",
        "luma": "observed_passthrough",
        "reference_manifest_sha256": manifest_hash,
        "degradations": reference["degradations"],
        "training": {key: value for key, value in settings.items()
                     if key not in {"device", "workers", "persistent_workers", "verify_hashes"}},
    }
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train_locked(reference, records, specs, manifest_bytes, manifest_hash,
                             dataset_root, output_dir, settings, contract, fingerprint,
                             device, workers, resume, stop_after_epoch)


def _train_locked(reference, records, specs, manifest_bytes, manifest_hash,
                  dataset_root, output_dir, settings, contract, fingerprint,
                  device, workers, resume, stop_after_epoch):
    epochs = int(settings["epochs"])
    seed = int(settings["seed"])
    crop_size = int(settings["crop_size"])
    batch_size = int(settings["batch_size"])
    last_path = output_dir / "last.pth"
    existing = [name for name in ("last.pth", "best.pth", "history.jsonl")
                if (output_dir / name).exists()]
    if existing and not resume:
        raise FileExistsError(f"V5 run already exists at {output_dir}: {existing}")
    if resume and not last_path.is_file():
        raise FileNotFoundError(f"Cannot resume without {last_path}")
    model = UNetGenerator().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(settings.get("min_learning_rate", 1e-6)),
    )
    checkpoint = None
    first_epoch, best_score, best_epoch = 1, math.inf, 0
    if resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        if checkpoint.get("fingerprint") != fingerprint:
            raise ValueError("Resume configuration or manifest differs from the original run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        first_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_chroma_mse"])
        best_epoch = int(checkpoint["best_epoch"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

    train_data = PrismDataset([r for r in records if r.split == "train"],
                              dataset_root, crop_size, specs, seed, training=True,
                              augment=settings.get("augment", True))
    val_data = PrismDataset([r for r in records if r.split == "validation"],
                            dataset_root, crop_size, specs, seed, training=False,
                            all_degradations=settings.get("validation_all_degradations", False))
    sampler = EpochSampler(train_data, seed)
    options = {"batch_size": batch_size, "num_workers": workers,
               "pin_memory": device.type == "cuda"}
    if workers:
        options["persistent_workers"] = settings.get("persistent_workers", True)
    train_loader = DataLoader(train_data, sampler=sampler,
                              generator=torch.Generator().manual_seed(seed), **options)
    val_loader = DataLoader(val_data, shuffle=False,
                            generator=torch.Generator().manual_seed(seed + 1), **options)
    history_path = output_dir / "history.jsonl"
    history = []
    if checkpoint:
        if history_path.exists():
            for line in history_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row["epoch"] < checkpoint["epoch"]:
                    history.append(row)
        history.append(checkpoint["metrics"])
    history_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                    for row in history))
    snapshot = {"reference": reference, "contract": contract,
                "manifest_sha256": manifest_hash, "dataset_root": str(dataset_root),
                "output_dir": str(output_dir)}
    atomic_json(output_dir / "config.snapshot.json", snapshot)
    (output_dir / "manifest.jsonl").write_bytes(manifest_bytes)
    print(json.dumps({"model": "v5_prism_mse", "first_epoch": first_epoch,
                      "epochs": epochs, "train_images": len(train_data.records),
                      "validation_images": len(val_data.records),
                      "output_dir": str(output_dir)}), flush=True)
    final_epoch = min(epochs, stop_after_epoch or epochs)
    for epoch in range(first_epoch, final_epoch + 1):
        sampler.set_epoch(epoch)
        model.train()
        total_mse = total_l1 = total_norm = 0.0
        samples = 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for inputs, targets, _, _, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = preserve_luma(inputs, model(inputs))
            mse = F.mse_loss(prediction[:, 1:3], targets[:, 1:3])
            if not torch.isfinite(mse):
                raise FloatingPointError("Non-finite training MSE")
            mse.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(settings.get("gradient_clip_norm", 1.0)),
                error_if_nonfinite=True,
            )
            optimizer.step()
            count = len(inputs)
            total_mse += float(mse.detach()) * count
            total_l1 += float(F.l1_loss(
                prediction[:, 1:3].detach(), targets[:, 1:3]
            )) * count
            total_norm += float(norm) * count
            samples += count
        if not samples:
            raise ValueError("Training has no samples")
        metrics, baseline, groups, validated = validate(model, val_loader, device)
        metrics["baseline"] = baseline
        metrics["samples"] = validated
        metrics["by_degradation"] = {
            specs[index].name: {"samples": group["samples"],
                                "chroma_mse": group["chroma_mse"] / group["samples"]}
            for index, group in groups.items()
        }
        row = {"model": "v5_prism_mse", "epoch": epoch, "epochs": epochs,
               "learning_rate": learning_rate,
               "training": {"chroma_mse": total_mse / samples,
                            "chroma_l1": total_l1 / samples,
                            "gradient_norm": total_norm / samples},
               "validation": metrics}
        score = metrics["chroma_mse"]
        improved = score < best_score
        if improved:
            best_score, best_epoch = score, epoch
        scheduler.step()
        payload = {"format_version": 1, "model_version": "v5",
                   "run_name": "v5_prism_mse", "epoch": epoch,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "fingerprint": fingerprint,
                   "best_chroma_mse": best_score, "best_epoch": best_epoch,
                   "metrics": row, "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        save_checkpoint(output_dir / f"epoch_{epoch:03d}.pth", payload)
        save_checkpoint(last_path, payload)
        if improved:
            save_checkpoint(output_dir / "best.pth", payload)
        with history_path.open("a") as output:
            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        history.append(row)
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    result = {"name": "v5_prism_mse", "completed_epochs": history[-1]["epoch"] if history else 0,
              "best_epoch": best_epoch, "best_chroma_mse": best_score,
              "best_checkpoint": str(output_dir / "best.pth"),
              "last_checkpoint": str(last_path)}
    atomic_json(output_dir / "summary.json", result)
    return result
