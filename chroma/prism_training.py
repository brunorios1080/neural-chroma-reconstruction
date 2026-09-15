"""One resumable, matched training/validation loop for every Prism experiment."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .prism_data import PrismDataset, EpochSampler, audit_manifest
from .prism_metrics import quality_batch
from .prism_models import PrismArchitecture, PrismRefiner, PrismDiscriminator
from .research_data import (
    load_manifest,
    DegradationSpec,
    SITING_OFFSETS,
    _downsample_matrix,
)
from .v7_losses import laplace_nll, von_mises_nll

FORMAT = "prism-v1"


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError(
            "Expected a Prism checkpoint; V5/V6/V7 checkpoints are incompatible"
        )
    model = PrismRefiner(PrismArchitecture(**payload["architecture"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


@lru_cache(maxsize=128)
def _matrices(height, width, spec, device):
    x, y = SITING_OFFSETS[spec.siting]
    return (
        torch.tensor(
            _downsample_matrix(height, y, spec.downsample_filter),
            device=device,
            dtype=torch.float32,
        ),
        torch.tensor(
            _downsample_matrix(width, x, spec.downsample_filter),
            device=device,
            dtype=torch.float32,
        ),
    )


def measurement_loss(chroma, observed_low, spec_indices, specs):
    total = chroma.new_zeros(())
    for index in torch.unique(spec_indices).tolist():
        mask = spec_indices == index
        selected = chroma[mask].float()
        wy, wx = _matrices(*selected.shape[-2:], specs[index], str(chroma.device))
        low = torch.einsum(
            "jw,bciw->bcij", wx, torch.einsum("ih,bchw->bciw", wy, selected)
        )
        total = total + F.l1_loss(low, observed_low[mask], reduction="sum")
    return total / observed_low.numel()


def ramp(epoch, warmup, duration):
    return max(0.0, min(1.0, (epoch - int(warmup)) / max(1, int(duration))))


def supervised_loss(
    prediction, target, low, indices, specs, config, epoch, architecture
):
    chroma, truth = prediction.image[:, 1:3], target[:, 1:3]
    l1 = F.l1_loss(chroma, truth)
    objective = config.get("reconstruction", "l1")
    if objective == "l1":
        reconstruction = l1
    elif objective == "mse":
        reconstruction = F.mse_loss(chroma, truth)
    elif objective == "charbonnier":
        reconstruction = ((chroma - truth).square() + 1e-6).sqrt().mean()
    else:
        raise ValueError(f"Unknown reconstruction objective: {objective}")
    edges = 0.5 * (
        F.l1_loss(chroma[..., 1:] - chroma[..., :-1], truth[..., 1:] - truth[..., :-1])
        + F.l1_loss(
            chroma[..., 1:, :] - chroma[..., :-1, :],
            truth[..., 1:, :] - truth[..., :-1, :],
        )
    )
    total = reconstruction + float(config.get("edge_weight", 0)) * edges
    values = {"chroma_l1": l1, "reconstruction": reconstruction, "gradient_l1": edges}
    weight = float(config.get("forward_weight", 0))
    if weight:
        forward = measurement_loss(chroma, low, indices, specs)
        total = total + weight * forward
        values["forward_l1"] = forward
    if prediction.scale is not None:
        centered = truth - 0.5
        amplitude = torch.linalg.vector_norm(centered, dim=1, keepdim=True)
        phase = torch.atan2(centered[:, 1:2], centered[:, 0:1])
        mean_a, mean_p = prediction.amplitude, prediction.phase
        if architecture.detached_uncertainty:
            mean_a, mean_p = mean_a.detach(), mean_p.detach()
        amplitude_nll = laplace_nll(amplitude, mean_a, prediction.scale).mean()
        weights = (amplitude / 0.05).clamp(0, 1)
        phase_nll = (
            weights * von_mises_nll(phase, mean_p, prediction.kappa)
        ).sum() / weights.sum().clamp_min(1e-6)
        factor = ramp(
            epoch,
            config.get("uncertainty_warmup_epochs", 5),
            config.get("uncertainty_ramp_epochs", 10),
        )
        if factor:
            total = total + factor * (
                float(config.get("amplitude_weight", 0.01)) * amplitude_nll
                + float(config.get("phase_weight", 0.005)) * phase_nll
            )
        values.update(
            amplitude_nll=amplitude_nll,
            phase_nll=phase_nll,
            uncertainty_ramp=total.new_tensor(factor),
        )
    values["total"] = total
    return total, values


def _limit(loader, maximum):
    for index, batch in enumerate(loader):
        if maximum and index >= maximum:
            break
        yield batch


@torch.no_grad()
def validate(model, loader, specs, device, max_batches=0):
    model.eval()
    total, baseline_total, groups = defaultdict(float), defaultdict(float), {}
    count = 0
    for inputs, targets, _, indices, _ in _limit(loader, max_batches):
        inputs, targets = inputs.to(device), targets.to(device)
        prediction = model.predict(inputs, [specs[i] for i in indices.tolist()])
        metrics = quality_batch(targets, prediction.image)
        baseline = quality_batch(targets, inputs)
        if prediction.scale is not None:
            for key, value in quality_batch(targets, prediction.safe_image()).items():
                metrics[f"safe_{key}"] = value
            target_a = torch.linalg.vector_norm(
                targets[:, 1:3] - 0.5, dim=1, keepdim=True
            )
            half_width = -prediction.scale * math.log(0.1)
            metrics["amplitude_coverage_90"] = (
                ((target_a - prediction.amplitude).abs() <= half_width)
                .float()
                .flatten(1)
                .mean(1)
            )
            metrics["amplitude_interval_width_90"] = (2 * half_width).flatten(1).mean(1)
        for key, value in metrics.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite validation metric: {key}")
            total[key] += float(value.sum())
        for key, value in baseline.items():
            baseline_total[key] += float(value.sum())
        for index in indices.unique().tolist():
            mask = (indices == index).to(device)
            group = groups.setdefault(
                specs[index].name,
                {
                    "samples": 0,
                    "chroma_l1_sum": 0.0,
                    "chroma_psnr_sum": 0.0,
                    "baseline_chroma_l1_sum": 0.0,
                },
            )
            group["samples"] += int(mask.sum())
            for key in ("chroma_l1", "chroma_psnr"):
                group[f"{key}_sum"] += float(metrics[key][mask].sum())
            group["baseline_chroma_l1_sum"] += float(baseline["chroma_l1"][mask].sum())
        count += len(inputs)
    if not count:
        raise ValueError("Validation has no usable samples")
    summary = {key: value / count for key, value in total.items()}
    summary["baseline"] = {key: value / count for key, value in baseline_total.items()}
    summary["chroma_l1_improvement"] = (
        summary["baseline"]["chroma_l1"] - summary["chroma_l1"]
    )
    summary["chroma_psnr_gain_db"] = (
        summary["chroma_psnr"] - summary["baseline"]["chroma_psnr"]
    )
    summary["samples"] = count
    summary["by_degradation"] = {
        name: {
            "samples": row["samples"],
            **{
                key.removesuffix("_sum"): value / row["samples"]
                for key, value in row.items()
                if key != "samples"
            },
        }
        for name, row in groups.items()
    }
    return summary


def _provenance(root):
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, text=True, capture_output=True, check=False
        ).stdout.strip()

    return {
        "git_revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "torch": str(torch.__version__),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / "chroma").glob("*.py")
        },
    }


def train_prism(config, project_root, resume=None, stop_after_epoch=None):
    root = Path(project_root).resolve()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else root / value
    destination = resolve(config["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another process owns this Prism run: {destination}"
            ) from error
        return _train(config, root, destination, resume, stop_after_epoch)


def _train(config, root, destination, resume, stop_after_epoch):
    resolve = lambda value: Path(value) if Path(value).is_absolute() else root / value
    training = config["training"]
    epochs, seed = int(training["epochs"]), int(training["seed"])
    batch_size, crop = int(training["batch_size"]), int(training["crop_size"])
    workers = int(training.get("workers", 0))
    if epochs < 1 or batch_size < 1 or workers < 0 or crop < 8 or crop % 2:
        raise ValueError("Invalid epoch/batch/worker/crop configuration")
    if stop_after_epoch is not None and stop_after_epoch < 1:
        raise ValueError("stop_after_epoch must be positive")
    if float(training.get("gradient_clip_norm", 1.0)) <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    if (
        not 0
        <= float(training.get("min_learning_rate", 1e-6))
        <= float(training["learning_rate"])
    ):
        raise ValueError("min_learning_rate must be between zero and learning_rate")
    for key in (
        "edge_weight",
        "forward_weight",
        "amplitude_weight",
        "phase_weight",
        "adversarial_weight",
    ):
        value = float(config["loss"].get(key, 0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid loss weight: {key}")
    if float(training["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    device = training.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = bool(training.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(training.get("allow_tf32", True))
    manifest = resolve(config["manifest"])
    records = load_manifest(manifest)
    dataset_root = resolve(config["dataset_root"])
    audit_manifest(
        records,
        dataset_root,
        config["data_kind"],
        crop,
        training.get("verify_hashes", True),
    )
    architecture = PrismArchitecture(**config["model"])
    architecture.validate()
    specs = [DegradationSpec(**entry) for entry in config["degradations"]]
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Degradation names must be unique")
    for spec in specs:
        spec.validate()
    portable_manifest = "".join(
        json.dumps(asdict(record), sort_keys=True) + "\n" for record in records
    )
    contract = {
        key: config[key]
        for key in ("name", "model", "loss", "data_kind", "degradations")
    }
    contract["training"] = {
        key: value
        for key, value in training.items()
        if key not in {"device", "workers", "persistent_workers", "verify_hashes"}
    }
    contract["manifest_sha256"] = hashlib.sha256(portable_manifest.encode()).hexdigest()
    fingerprint = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()
    last_path = destination / "last.pth"
    resume_path = last_path if resume == "auto" else (Path(resume) if resume else None)
    if resume == "auto" and not resume_path.exists():
        resume_path = None
    if resume_path is not None and resume_path.resolve() != last_path.resolve():
        raise ValueError(
            "Resume must use this output directory's last.pth. To relocate a run, "
            "copy its entire directory, including best.pth and manifest.jsonl."
        )
    if resume_path is None and any(
        (destination / name).exists()
        for name in ("last.pth", "best.pth", "config.snapshot.json", "history.jsonl")
    ):
        raise FileExistsError(
            f"Run already exists at {destination}; use --resume auto or a new output root"
        )
    model = PrismRefiner(architecture).to(device)
    adv_weight = float(config["loss"].get("adversarial_weight", 0))
    discriminator = (
        PrismDiscriminator(int(config["loss"].get("discriminator_width", 32))).to(
            device
        )
        if adv_weight
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(training.get("min_learning_rate", 1e-6))
    )
    d_optimizer = (
        torch.optim.AdamW(
            discriminator.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training.get("weight_decay", 1e-4)),
        )
        if discriminator
        else None
    )
    d_schedule = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            d_optimizer,
            T_max=epochs,
            eta_min=float(training.get("min_learning_rate", 1e-6)),
        )
        if discriminator
        else None
    )
    first_epoch, best_score, best_epoch, checkpoint = 1, float("inf"), 0, None
    if resume_path is not None:
        _, checkpoint = load_checkpoint(resume_path, "cpu")
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError(
                "Resume configuration/manifest differs from checkpoint; use a new run for changed experiments"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        schedule.load_state_dict(checkpoint["scheduler"])
        if discriminator:
            discriminator.load_state_dict(checkpoint["discriminator"])
            d_optimizer.load_state_dict(checkpoint["d_optimizer"])
            d_schedule.load_state_dict(checkpoint["d_scheduler"])
        first_epoch = checkpoint["epoch"] + 1
        best_score, best_epoch = checkpoint["best_score"], checkpoint["best_epoch"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

    def dataset(split, training_mode):
        return PrismDataset(
            [r for r in records if r.split == split],
            dataset_root,
            crop,
            specs,
            seed,
            training=training_mode,
            all_degradations=not training_mode
            and training.get("validation_all_degradations", False),
            augment=training.get("augment", True),
        )

    train_data, val_data = dataset("train", True), dataset("validation", False)
    sampler = EpochSampler(train_data, seed)
    options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers:
        options["persistent_workers"] = training.get("persistent_workers", True)
    # Separate worker RNG from model RNG; sampling itself is epoch-addressed.
    train_loader = DataLoader(
        train_data,
        sampler=sampler,
        generator=torch.Generator().manual_seed(seed),
        **options,
    )
    val_loader = DataLoader(
        val_data,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 1),
        **options,
    )
    history_path = destination / "history.jsonl"
    history = []
    if checkpoint and history_path.exists():
        for line in history_path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record["epoch"] < checkpoint["epoch"]:
                history.append(record)
    if checkpoint:
        history.append(checkpoint["metrics"])
    # Recover an interrupted append using the last atomic checkpoint as authority.
    history_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history)
    )
    atomic_json(destination / "config.snapshot.json", config)
    atomic_json(destination / "provenance.json", _provenance(root))
    (destination / "manifest.jsonl").write_text(portable_manifest)
    print(
        json.dumps(
            {
                "model": config["name"],
                "device": str(device),
                "parameters": sum(p.numel() for p in model.parameters()),
                "train_images": len(train_data.records),
                "validation_images": len(val_data.records),
                "first_epoch": first_epoch,
                "epochs": epochs,
                "output_dir": str(destination),
            }
        ),
        flush=True,
    )
    last_epoch = (
        min(epochs, stop_after_epoch) if stop_after_epoch is not None else epochs
    )
    for epoch in range(first_epoch, last_epoch + 1):
        sampler.set_epoch(epoch)
        model.train()
        totals, samples = defaultdict(float), 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for inputs, target, low, indices, _ in _limit(
            train_loader, int(training.get("max_train_batches", 0))
        ):
            inputs, target, low, indices = (
                x.to(device, non_blocking=True) for x in (inputs, target, low, indices)
            )
            observation_specs = [specs[index] for index in indices.tolist()]
            prediction = model.predict(inputs, observation_specs)
            factor = ramp(
                epoch,
                config["loss"].get("adversarial_warmup_epochs", 5),
                config["loss"].get("adversarial_ramp_epochs", 5),
            )
            d_loss = target.new_zeros(())
            if discriminator and factor:
                discriminator.train()
                d_optimizer.zero_grad(set_to_none=True)
                real = discriminator(inputs, target)
                fake = discriminator(inputs, prediction.image.detach())
                d_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
                    + F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
                )
                if not torch.isfinite(d_loss):
                    raise FloatingPointError("Non-finite discriminator loss")
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(),
                    float(training.get("gradient_clip_norm", 1.0)),
                    error_if_nonfinite=True,
                )
                d_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss, components = supervised_loss(
                prediction,
                target,
                low,
                indices,
                specs,
                config["loss"],
                epoch,
                architecture,
            )
            if discriminator and factor:
                discriminator.requires_grad_(False)
                logits = discriminator(inputs, prediction.image)
                adversarial = F.binary_cross_entropy_with_logits(
                    logits, torch.ones_like(logits)
                )
                loss = loss + adv_weight * factor * adversarial
                components.update(adversarial=adversarial, discriminator=d_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Prism training loss")
            loss.backward()
            clip = float(training.get("gradient_clip_norm", 1.0))
            if architecture.detached_uncertainty:
                # A combined norm would let large auxiliary gradients rescale the
                # reconstruction update even though its graph is detached.
                main_parameters = [
                    p
                    for name, p in model.named_parameters()
                    if not name.startswith("uncertainty_head.")
                ]
                norm = torch.nn.utils.clip_grad_norm_(
                    main_parameters, clip, error_if_nonfinite=True
                )
                aux_norm = torch.nn.utils.clip_grad_norm_(
                    model.uncertainty_head.parameters(), clip, error_if_nonfinite=True
                )
                components["uncertainty_gradient_norm"] = aux_norm
            else:
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip, error_if_nonfinite=True
                )
            optimizer.step()
            if discriminator:
                discriminator.requires_grad_(True)
            components.update(total=loss, gradient_norm=norm)
            for key, value in components.items():
                totals[key] += float(value.detach()) * len(inputs)
            samples += len(inputs)
        if not samples:
            raise ValueError("Training has no samples")
        validation = validate(
            model,
            val_loader,
            specs,
            device,
            int(training.get("max_validation_batches", 0)),
        )
        row = {
            "model": config["name"],
            "epoch": epoch,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "training": {key: value / samples for key, value in totals.items()},
            "validation": validation,
        }
        score = validation["chroma_l1"]
        improved = score < best_score
        if improved:
            best_score, best_epoch = score, epoch
        schedule.step()
        if d_schedule and factor:
            d_schedule.step()
        payload = {
            "format": FORMAT,
            "name": config["name"],
            "architecture": asdict(architecture),
            "config": config,
            "fingerprint": fingerprint,
            "epoch": epoch,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": schedule.state_dict(),
            "metrics": row,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        if discriminator:
            payload.update(
                discriminator=discriminator.state_dict(),
                d_optimizer=d_optimizer.state_dict(),
                d_scheduler=d_schedule.state_dict(),
            )
        if improved:
            save_checkpoint(destination / "best.pth", payload)
        save_checkpoint(last_path, payload)
        with history_path.open("a") as output:
            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        history.append(row)
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    result = {
        "name": config["name"],
        "best_epoch": best_epoch,
        "best_chroma_l1": best_score,
        "best_checkpoint": str(destination / "best.pth"),
        "last_checkpoint": str(last_path),
        "completed_epochs": history[-1]["epoch"] if history else 0,
    }
    atomic_json(destination / "summary.json", result)
    return result
