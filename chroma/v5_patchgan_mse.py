"""Fresh V5 PatchGAN training on the completed Prism MSE data protocol."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import Discriminator, UNetGenerator
from .prism_data import EpochSampler, PrismDataset, audit_manifest
from .prism_metrics import quality_batch
from .prism_training import atomic_json, save_checkpoint
from .research_data import DegradationSpec, load_manifest

RUN_NAME = "v5_patchgan_mse100"
GENERATOR_LR = 2e-4
DISCRIMINATOR_LR = 2e-4
RECONSTRUCTION_WEIGHT = 10.0


def mse_per_image(target, prediction, chroma_only=False):
    if chroma_only:
        target, prediction = target[:, 1:3], prediction[:, 1:3]
    return (target - prediction).square().flatten(1).mean(1)


@torch.no_grad()
def validate(generator, loader, device, specs):
    generator.eval()
    totals, baseline_totals, groups = defaultdict(float), defaultdict(float), {}
    samples = 0
    for inputs, targets, _, indices, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        prediction = generator(inputs)
        metrics = quality_batch(targets, prediction)
        baseline = quality_batch(targets, inputs)
        for name, chroma_only in (("full_mse", False), ("chroma_mse", True)):
            metrics[name] = mse_per_image(targets, prediction, chroma_only)
            baseline[name] = mse_per_image(targets, inputs, chroma_only)
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
    result = {key: value / samples for key, value in totals.items()}
    result["baseline"] = {
        key: value / samples for key, value in baseline_totals.items()
    }
    result["samples"] = samples
    result["by_degradation"] = {
        specs[index].name: {
            "samples": row["samples"],
            "chroma_mse": row["chroma_mse"] / row["samples"],
        }
        for index, row in groups.items()
    }
    return result


def load_patchgan(path, device="cpu"):
    """Load the raw V5 generator and PatchGAN state from this run."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("run_name") != RUN_NAME or "discriminator" not in payload:
        raise ValueError("Expected a V5 PatchGAN MSE checkpoint")
    generator = UNetGenerator()
    generator.load_state_dict(payload["model"], strict=True)
    discriminator = Discriminator()
    discriminator.load_state_dict(payload["discriminator"], strict=True)
    return generator.to(device).eval(), discriminator.to(device).eval(), payload


def train(reference_run, dataset_root, output_dir, batch_size=64, workers=5,
          resume=False, stop_after_epoch=None, device_name="cuda"):
    reference_run = Path(reference_run).resolve()
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == reference_run:
        raise ValueError("V5 output must be separate from the Prism reference")
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers nonnegative")
    reference = json.loads((reference_run / "config.snapshot.json").read_text())
    if reference["loss"]["reconstruction"] != "mse":
        raise ValueError("Reference Prism run must use MSE reconstruction")
    manifest_bytes = (reference_run / "manifest.jsonl").read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    records = load_manifest(reference_run / "manifest.jsonl")
    settings = reference["training"]
    epochs, crop_size, seed = (
        int(settings[key]) for key in ("epochs", "crop_size", "seed")
    )
    if epochs != 100 or crop_size < 64 or crop_size % 8:
        raise ValueError("Expected 100 epochs and a V5-compatible Prism crop")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= epochs:
        raise ValueError("stop_after_epoch must be within the run")
    specs = [DegradationSpec(**entry) for entry in reference["degradations"]]
    for spec in specs:
        spec.validate()
    # The new job stages its own archive, so verify it against the saved split.
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
        "model": "v5_unet_generator_and_patchgan_discriminator",
        "loss": "bce_adversarial_plus_10x_full_image_mse",
        "checkpoint_selection": "validation_chroma_mse",
        "generator_lr": GENERATOR_LR,
        "discriminator_lr": DISCRIMINATOR_LR,
        "optimizer": "adam_betas_0.5_0.999",
        "batch_size": batch_size,
        "reference_manifest_sha256": manifest_hash,
        "degradations": reference["degradations"],
        "training": {
            key: value for key, value in settings.items()
            if key not in {"batch_size", "device", "workers", "persistent_workers", "verify_hashes"}
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train_locked(reference, records, specs, manifest_bytes,
                             manifest_hash, dataset_root, output_dir, settings,
                             contract, fingerprint, device, batch_size, workers,
                             resume, stop_after_epoch)


def _train_locked(reference, records, specs, manifest_bytes, manifest_hash,
                  dataset_root, output_dir, settings, contract, fingerprint,
                  device, batch_size, workers, resume, stop_after_epoch):
    epochs, crop_size, seed = (
        int(settings[key]) for key in ("epochs", "crop_size", "seed")
    )
    last_path = output_dir / "last.pth"
    existing = [name for name in ("last.pth", "best.pth", "history.jsonl")
                if (output_dir / name).exists()]
    if existing and not resume:
        raise FileExistsError(f"V5 PatchGAN run already exists: {existing}")
    if resume and not last_path.is_file():
        raise FileNotFoundError(f"Cannot resume without {last_path}")
    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=GENERATOR_LR, betas=(0.5, 0.999)
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=DISCRIMINATOR_LR, betas=(0.5, 0.999)
    )
    checkpoint = None
    first_epoch, best_score, best_epoch = 1, math.inf, 0
    if resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        if checkpoint.get("fingerprint") != fingerprint:
            raise ValueError("Resume configuration or manifest differs")
        generator.load_state_dict(checkpoint["model"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        g_optimizer.load_state_dict(checkpoint["optimizers"]["generator"])
        d_optimizer.load_state_dict(checkpoint["optimizers"]["discriminator"])
        first_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_chroma_mse"])
        best_epoch = int(checkpoint["best_epoch"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    train_data = PrismDataset(
        [r for r in records if r.split == "train"], dataset_root, crop_size,
        specs, seed, training=True, augment=settings.get("augment", True)
    )
    val_data = PrismDataset(
        [r for r in records if r.split == "validation"], dataset_root, crop_size,
        specs, seed, training=False,
        all_degradations=settings.get("validation_all_degradations", False),
    )
    sampler = EpochSampler(train_data, seed)
    options = {"batch_size": batch_size, "num_workers": workers,
               "pin_memory": device.type == "cuda"}
    if workers:
        options["persistent_workers"] = settings.get("persistent_workers", True)
    train_loader = DataLoader(
        train_data, sampler=sampler,
        generator=torch.Generator().manual_seed(seed), **options
    )
    val_loader = DataLoader(
        val_data, shuffle=False,
        generator=torch.Generator().manual_seed(seed + 1), **options
    )
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
    history_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history)
    )
    atomic_json(output_dir / "config.snapshot.json", {
        "reference": reference, "contract": contract,
        "manifest_sha256": manifest_hash, "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
    })
    (output_dir / "manifest.jsonl").write_bytes(manifest_bytes)
    print(json.dumps({
        "model": RUN_NAME, "first_epoch": first_epoch, "epochs": epochs,
        "batch_size": batch_size, "train_images": len(train_data.records),
        "validation_images": len(val_data.records), "output_dir": str(output_dir),
    }), flush=True)
    final_epoch = min(epochs, stop_after_epoch or epochs)
    for epoch in range(first_epoch, final_epoch + 1):
        sampler.set_epoch(epoch)
        generator.train()
        discriminator.train()
        totals = defaultdict(float)
        samples = 0
        for inputs, targets, _, _, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            d_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                detached_prediction = generator(inputs)
            real = discriminator(targets)
            fake = discriminator(detached_prediction)
            d_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
                + F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
            )
            if not torch.isfinite(d_loss):
                raise FloatingPointError("Non-finite PatchGAN discriminator loss")
            d_loss.backward()
            d_optimizer.step()
            discriminator.requires_grad_(False)
            g_optimizer.zero_grad(set_to_none=True)
            prediction = generator(inputs)
            logits = discriminator(prediction)
            adversarial = F.binary_cross_entropy_with_logits(
                logits, torch.ones_like(logits)
            )
            full_mse = F.mse_loss(prediction, targets)
            g_loss = adversarial + RECONSTRUCTION_WEIGHT * full_mse
            if not torch.isfinite(g_loss):
                raise FloatingPointError("Non-finite PatchGAN generator loss")
            g_loss.backward()
            g_optimizer.step()
            discriminator.requires_grad_(True)
            count = len(inputs)
            metrics = {
                "generator_loss": g_loss,
                "discriminator_loss": d_loss,
                "adversarial_loss": adversarial,
                "full_mse": full_mse,
                "chroma_mse": F.mse_loss(prediction[:, 1:3], targets[:, 1:3]),
                "chroma_l1": F.l1_loss(prediction[:, 1:3], targets[:, 1:3]),
            }
            for name, value in metrics.items():
                totals[name] += float(value.detach()) * count
            samples += count
        if not samples:
            raise ValueError("Training has no samples")
        validation = validate(generator, val_loader, device, specs)
        row = {
            "model": RUN_NAME, "epoch": epoch, "epochs": epochs,
            "training": {name: value / samples for name, value in totals.items()},
            "validation": validation,
        }
        score = validation["chroma_mse"]
        improved = score < best_score
        if improved:
            best_score, best_epoch = score, epoch
        payload = {
            "format_version": 1, "model_version": "v5", "run_name": RUN_NAME,
            "epoch": epoch, "model": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizers": {
                "generator": g_optimizer.state_dict(),
                "discriminator": d_optimizer.state_dict(),
            },
            "fingerprint": fingerprint, "best_chroma_mse": best_score,
            "best_epoch": best_epoch, "metrics": row,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if device.type == "cuda" else [],
        }
        save_checkpoint(output_dir / f"epoch_{epoch:03d}.pth", payload)
        save_checkpoint(last_path, payload)
        if improved:
            save_checkpoint(output_dir / "best.pth", payload)
        with history_path.open("a") as output:
            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        history.append(row)
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    result = {
        "name": RUN_NAME,
        "completed_epochs": history[-1]["epoch"] if history else 0,
        "best_epoch": best_epoch, "best_chroma_mse": best_score,
        "best_checkpoint": str(output_dir / "best.pth"),
        "last_checkpoint": str(last_path),
    }
    atomic_json(output_dir / "summary.json", result)
    return result
