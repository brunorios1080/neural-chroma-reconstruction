"""Image discovery and synthetic 4:2:0 dataset utilities."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_image_files(source: str | Path) -> list[Path]:
    root = Path(source)
    if not root.exists():
        raise FileNotFoundError(f"Image source does not exist: {root}")
    files = sorted(
        path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No supported images found under: {root}")
    return files


def split_files(
    files: Sequence[Path], val_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in the range [0, 1)")
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    if val_fraction == 0.0 or len(shuffled) == 1:
        return shuffled, []
    val_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * val_fraction)))
    return shuffled[val_count:], shuffled[:val_count]


def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_ycrcb(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0


def ycrcb_to_bgr_uint8(ycrcb: np.ndarray) -> np.ndarray:
    clipped = np.clip(ycrcb, 0.0, 1.0)
    ycrcb_u8 = np.rint(clipped * 255.0).astype(np.uint8)
    return cv2.cvtColor(ycrcb_u8, cv2.COLOR_YCrCb2BGR)


def simulate_420(
    ycrcb: np.ndarray, interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """Return Y plus chroma downsampled to 4:2:0 and restored to full size."""
    height, width = ycrcb.shape[:2]
    low_size = (max(1, (width + 1) // 2), max(1, (height + 1) // 2))
    channels = [ycrcb[:, :, 0:1]]
    for index in (1, 2):
        low = cv2.resize(ycrcb[:, :, index], low_size, interpolation=cv2.INTER_AREA)
        restored = cv2.resize(low, (width, height), interpolation=interpolation)
        channels.append(restored[:, :, None])
    return np.concatenate(channels, axis=2).astype(np.float32, copy=False)


def to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)


class YUVChromaDataset(Dataset):
    def __init__(
        self,
        files: Sequence[str | Path],
        crop_size: int | None,
        random_crop: bool,
    ) -> None:
        self.files = [Path(path) for path in files]
        self.crop_size = crop_size
        self.random_crop = random_crop
        if crop_size is not None and crop_size < 2:
            raise ValueError("crop_size must be at least 2")

    def __len__(self) -> int:
        return len(self.files)

    def _crop(self, image: np.ndarray) -> np.ndarray:
        if self.crop_size is None:
            return image
        height, width = image.shape[:2]
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(
                f"image is {width}x{height}, smaller than {self.crop_size}x{self.crop_size} crop"
            )
        if self.random_crop:
            top = random.randint(0, height - self.crop_size)
            left = random.randint(0, width - self.crop_size)
        else:
            top = (height - self.crop_size) // 2
            left = (width - self.crop_size) // 2
        return image[top : top + self.crop_size, left : left + self.crop_size]

    def __getitem__(self, index: int):
        path = self.files[index]
        try:
            target = rgb_to_ycrcb(self._crop(read_rgb(path)))
            model_input = simulate_420(target)
            return to_tensor(model_input), to_tensor(target), str(path)
        except (OSError, ValueError, cv2.error) as error:
            return None, str(path), str(error)


def collate_valid(batch):
    valid = [sample for sample in batch if sample[0] is not None]
    failures = [sample[1:] for sample in batch if sample[0] is None]
    if not valid:
        return None, failures
    inputs, targets, paths = zip(*valid)
    return (torch.stack(inputs), torch.stack(targets), list(paths)), failures


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
