"""Matched epoch-addressed crops, including persistent-worker-safe sampling."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image

from .research_data import (
    LOSSLESS_FORMATS,
    FULL_COLOR_MODES,
    read_rgb,
    rgb_to_ycrcb,
    simulate_420,
    sha256_file,
)


def audit_manifest(records, root, data_kind, crop_size, verify_hashes=True):
    if data_kind not in {"lossless", "synthetic_coco"}:
        raise ValueError("data_kind must be lossless or synthetic_coco")
    root = Path(root).resolve()
    seen_ids, seen_paths, hashes = set(), set(), {}
    for record in records:
        path = (root / record.relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or out-of-root manifest image: {record.id}")
        if record.id in seen_ids or path in seen_paths:
            raise ValueError(f"Duplicate manifest id/path: {record.id}")
        seen_ids.add(record.id)
        seen_paths.add(path)
        if record.sha256 in hashes and hashes[record.sha256] != record.split:
            raise ValueError(f"Cross-split duplicate content: {record.id}")
        hashes[record.sha256] = record.split
        if min(record.width, record.height) < crop_size:
            raise ValueError(f"Image smaller than crop_size: {record.id}")
        if verify_hashes and sha256_file(path) != record.sha256:
            raise ValueError(f"Manifest hash mismatch: {record.id}")
        if data_kind == "lossless":
            with Image.open(path) as image:
                if (
                    image.format not in LOSSLESS_FORMATS
                    or image.mode not in FULL_COLOR_MODES
                ):
                    raise ValueError(
                        f"Lossless full-color source required: {record.id}"
                    )


class EpochSampler(Sampler):
    """Put epoch in each worker task instead of mutating a worker-owned dataset."""

    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, int(seed), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        for index in torch.randperm(len(self.dataset), generator=generator).tolist():
            yield self.epoch, index

    def __len__(self):
        return len(self.dataset)


class PrismDataset(Dataset):
    def __init__(
        self,
        records,
        root,
        crop_size,
        specs,
        seed,
        training=False,
        all_degradations=False,
        augment=True,
    ):
        if crop_size < 8 or crop_size % 2 or not records or not specs:
            raise ValueError(
                "Prism requires records, degradations, and an even crop_size >= 8"
            )
        self.records, self.root, self.crop_size = list(records), Path(root), crop_size
        self.specs, self.seed = list(specs), seed
        self.training, self.all_degradations, self.augment = (
            training,
            all_degradations,
            augment,
        )

    def __len__(self):
        return len(self.records) * (len(self.specs) if self.all_degradations else 1)

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        record_index = index // len(self.specs) if self.all_degradations else index
        record = self.records[record_index]
        image = read_rgb(self.root / record.relative_path)
        height, width = image.shape[:2]
        if min(height, width) < self.crop_size:
            raise ValueError(f"Image smaller than crop_size: {record.id}")
        digest = hashlib.sha256(f"{self.seed}:{epoch}:{record.id}".encode()).digest()
        random = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        if self.training:
            top = int(random.integers(height - self.crop_size + 1))
            left = int(random.integers(width - self.crop_size + 1))
            spec_index = int(random.integers(len(self.specs)))
        else:
            top, left = (height - self.crop_size) // 2, (width - self.crop_size) // 2
            spec_index = index % len(self.specs)
        crop = image[top : top + self.crop_size, left : left + self.crop_size]
        # Transform the source BEFORE degradation so cosited sampling stays valid.
        if self.training and self.augment:
            crop = np.rot90(crop, int(random.integers(4)))
            if random.integers(2):
                crop = np.flip(crop, axis=1)
        target = rgb_to_ycrcb(np.ascontiguousarray(crop))
        observed, low = simulate_420(target, self.specs[spec_index])
        tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
        return tensor(observed), tensor(target), tensor(low), spec_index, record.id
