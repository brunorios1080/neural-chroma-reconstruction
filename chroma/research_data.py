"""Manifest and color-sampling utilities for publication-grade experiments.

This module deliberately avoids OpenCV so the research protocol can be run in a
minimal NumPy/Pillow environment.  Images are represented as normalized RGB or
YCrCb arrays in channel-last order.  Chroma is ordered as Cr, Cb to remain
compatible with the historical checkpoints in this repository.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

LOSSLESS_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".ppm", ".pgm"}
LOSSLESS_FORMATS = {"PNG", "TIFF", "BMP", "PPM"}
FULL_COLOR_MODES = {"RGB", "RGBA"}
SITING_OFFSETS = {
    "center": (0.5, 0.5),
    "left": (0.0, 0.5),
    "cosited": (0.0, 0.0),
}
DOWNSAMPLE_FILTERS = {"box", "triangle", "gaussian", "lanczos3", "point"}
UPSAMPLE_FILTERS = {"nearest", "bilinear", "bicubic", "lanczos3"}


@dataclass(frozen=True)
class ManifestRecord:
    id: str
    relative_path: str
    split: str
    sha256: str
    width: int
    height: int
    mode: str
    image_format: str
    source_group: str


@dataclass(frozen=True)
class DegradationSpec:
    name: str
    siting: str = "center"
    downsample_filter: str = "box"
    upsample_filter: str = "bilinear"

    def validate(self) -> None:
        if self.siting not in SITING_OFFSETS:
            raise ValueError(
                f"Unknown chroma siting {self.siting!r}; "
                f"choose from {sorted(SITING_OFFSETS)}"
            )
        if self.downsample_filter not in DOWNSAMPLE_FILTERS:
            raise ValueError(
                f"Unknown downsampling filter {self.downsample_filter!r}; "
                f"choose from {sorted(DOWNSAMPLE_FILTERS)}"
            )
        if self.upsample_filter not in UPSAMPLE_FILTERS:
            raise ValueError(
                f"Unknown upsampling filter {self.upsample_filter!r}; "
                f"choose from {sorted(UPSAMPLE_FILTERS)}"
            )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return rgb


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(destination)


def rgb_to_ycrcb(rgb: np.ndarray) -> np.ndarray:
    """Convert normalized sRGB to full-range BT.601-style Y, Cr, Cb."""
    rgb = np.asarray(rgb, dtype=np.float32)
    red, green, blue = (rgb[..., index] for index in range(3))
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    cr = 0.5 + 0.713 * (red - luma)
    cb = 0.5 + 0.564 * (blue - luma)
    return np.stack((luma, cr, cb), axis=-1).astype(np.float32)


def ycrcb_to_rgb(ycrcb: np.ndarray) -> np.ndarray:
    ycrcb = np.asarray(ycrcb, dtype=np.float32)
    luma = ycrcb[..., 0]
    cr = ycrcb[..., 1] - 0.5
    cb = ycrcb[..., 2] - 0.5
    red = luma + 1.403 * cr
    green = luma - 0.714 * cr - 0.344 * cb
    blue = luma + 1.773 * cb
    return np.clip(np.stack((red, green, blue), axis=-1), 0.0, 1.0)


def _image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def build_manifest(
    split_directories: Mapping[str, str | Path],
    output_path: str | Path,
    dataset_root: str | Path | None = None,
    allow_lossy: bool = False,
) -> list[ManifestRecord]:
    """Create a hash-addressed JSONL manifest and reject split leakage.

    The manifest stores paths relative to ``dataset_root`` when supplied.  A
    SHA-256 collision across splits is treated as data leakage and aborts the
    build.  By default only lossless raster formats are admitted.
    """
    if not split_directories:
        raise ValueError("At least one split directory is required")
    root = Path(dataset_root).resolve() if dataset_root is not None else None
    records: list[ManifestRecord] = []
    hashes: dict[str, tuple[str, Path]] = {}
    ids: set[str] = set()
    skipped: list[dict[str, str]] = []

    for split, directory_value in sorted(split_directories.items()):
        directory = Path(directory_value).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Split directory does not exist: {directory}")
        for path in _image_files(directory):
            metadata_path = f"{split}/{path.relative_to(directory).as_posix()}"
            if not allow_lossy and path.suffix.lower() not in LOSSLESS_EXTENSIONS:
                skipped.append({"path": metadata_path, "reason": "not lossless"})
                continue
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    mode = image.mode
                    image_format = (image.format or path.suffix.lstrip(".")).upper()
                    image.verify()
            except Exception as error:
                skipped.append({"path": metadata_path, "reason": str(error)})
                continue
            if not allow_lossy and image_format not in LOSSLESS_FORMATS:
                skipped.append(
                    {
                        "path": metadata_path,
                        "reason": f"format {image_format} is not lossless",
                    }
                )
                continue
            if mode not in FULL_COLOR_MODES:
                skipped.append(
                    {
                        "path": metadata_path,
                        "reason": f"mode {mode} is not explicit RGB/RGBA 4:4:4",
                    }
                )
                continue
            if width < 2 or height < 2:
                skipped.append({"path": metadata_path, "reason": "smaller than 2x2"})
                continue
            digest = sha256_file(path)
            previous = hashes.get(digest)
            if previous is not None and previous[0] != split:
                raise ValueError(
                    "Training/test leakage: identical bytes occur in splits "
                    f"{previous[0]!r} ({previous[1]}) and {split!r} ({path})"
                )
            hashes[digest] = (split, path)
            if root is not None:
                try:
                    relative = path.relative_to(root).as_posix()
                except ValueError as error:
                    raise ValueError(f"{path} is outside dataset root {root}") from error
            else:
                relative = path.relative_to(directory.parent).as_posix()
            record_id = f"{split}/{path.relative_to(directory).with_suffix('').as_posix()}"
            if record_id in ids:
                raise ValueError(f"Duplicate manifest id: {record_id}")
            ids.add(record_id)
            records.append(
                ManifestRecord(
                    id=record_id,
                    relative_path=relative,
                    split=split,
                    sha256=digest,
                    width=width,
                    height=height,
                    mode=mode,
                    image_format=image_format,
                    source_group=path.parent.name,
                )
            )

    if not records:
        raise ValueError("No eligible images found; lossless sources are required")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    if root is None:
        portable_root = None
    else:
        try:
            portable_root = Path(
                os.path.relpath(root, start=destination.parent.resolve())
            ).as_posix()
        except ValueError:
            portable_root = root.name
    metadata = {
        "format_version": 1,
        "records": len(records),
        "splits": {
            split: sum(record.split == split for record in records)
            for split in sorted({record.split for record in records})
        },
        "lossless_only": not allow_lossy,
        "allowed_formats": sorted(LOSSLESS_FORMATS),
        "allowed_modes": sorted(FULL_COLOR_MODES),
        "dataset_root": portable_root,
        "manifest_sha256": sha256_file(destination),
        "skipped": skipped,
    }
    destination.with_suffix(destination.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return records


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    records = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(ManifestRecord(**json.loads(line)))
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid manifest record on line {line_number}") from error
    if not records:
        raise ValueError(f"Manifest contains no records: {path}")
    return records


def verify_manifest(
    records: Sequence[ManifestRecord], dataset_root: str | Path
) -> dict[str, object]:
    root = Path(dataset_root)
    failures = []
    split_hashes: dict[str, set[str]] = {}
    for record in records:
        split_hashes.setdefault(record.split, set()).add(record.sha256)
        path = root / record.relative_path
        if not path.is_file():
            failures.append({"id": record.id, "reason": "missing", "path": str(path)})
            continue
        actual = sha256_file(path)
        if actual != record.sha256:
            failures.append(
                {"id": record.id, "reason": "sha256 mismatch", "path": str(path)}
            )
    split_names = sorted(split_hashes)
    leakage = []
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            overlap = split_hashes[first] & split_hashes[second]
            if overlap:
                leakage.append(
                    {"splits": [first, second], "duplicate_hashes": sorted(overlap)}
                )
    return {
        "ok": not failures and not leakage,
        "records": len(records),
        "failures": failures,
        "split_leakage": leakage,
    }


def _kernel_weights(distance: np.ndarray, name: str) -> np.ndarray:
    absolute = np.abs(distance)
    if name == "point":
        weights = (absolute == np.min(absolute, axis=1, keepdims=True)).astype(
            np.float64
        )
    elif name == "box":
        weights = (absolute <= 0.5000001).astype(np.float64)
    elif name == "triangle":
        weights = np.maximum(0.0, 1.0 - absolute / 2.0)
    elif name == "gaussian":
        weights = np.exp(-0.5 * (distance / 0.85) ** 2) * (absolute <= 2.75)
    elif name == "lanczos3":
        scaled = distance / 2.0
        weights = np.sinc(scaled) * np.sinc(scaled / 3.0) * (np.abs(scaled) < 3.0)
    elif name == "nearest":
        weights = (absolute == np.min(absolute, axis=1, keepdims=True)).astype(
            np.float64
        )
    elif name == "bilinear":
        weights = np.maximum(0.0, 1.0 - absolute)
    elif name == "bicubic":
        a = -0.5
        weights = np.zeros_like(distance, dtype=np.float64)
        first = absolute <= 1.0
        second = (absolute > 1.0) & (absolute < 2.0)
        weights[first] = (
            (a + 2.0) * absolute[first] ** 3
            - (a + 3.0) * absolute[first] ** 2
            + 1.0
        )
        weights[second] = (
            a * absolute[second] ** 3
            - 5.0 * a * absolute[second] ** 2
            + 8.0 * a * absolute[second]
            - 4.0 * a
        )
    else:
        raise ValueError(f"Unsupported resampling kernel: {name}")
    totals = np.sum(weights, axis=1, keepdims=True)
    zero_rows = totals[:, 0] <= 1e-12
    if np.any(zero_rows):
        nearest = np.argmin(absolute[zero_rows], axis=1)
        weights[zero_rows] = 0.0
        weights[np.flatnonzero(zero_rows), nearest] = 1.0
        totals = np.sum(weights, axis=1, keepdims=True)
    return weights / totals


def _downsample_matrix(input_size: int, offset: float, filter_name: str) -> np.ndarray:
    if input_size % 2:
        raise ValueError("Research crops must have even width and height for 4:2:0")
    output_size = input_size // 2
    sample_positions = 2.0 * np.arange(output_size, dtype=np.float64) + offset
    source_positions = np.arange(input_size, dtype=np.float64)
    distance = source_positions[None, :] - sample_positions[:, None]
    return _kernel_weights(distance, filter_name)


def _upsample_matrix(
    low_size: int, output_size: int, offset: float, filter_name: str
) -> np.ndarray:
    sample_positions = 2.0 * np.arange(low_size, dtype=np.float64) + offset
    output_positions = np.arange(output_size, dtype=np.float64)
    distance_in_low_samples = (
        output_positions[:, None] - sample_positions[None, :]
    ) / 2.0
    return _kernel_weights(distance_in_low_samples, filter_name)


def downsample_chroma(
    chroma: np.ndarray, siting: str = "center", filter_name: str = "box"
) -> np.ndarray:
    if siting not in SITING_OFFSETS:
        raise ValueError(f"Unsupported chroma siting: {siting}")
    if filter_name not in DOWNSAMPLE_FILTERS:
        raise ValueError(f"Unsupported downsampling filter: {filter_name}")
    height, width, channels = chroma.shape
    if channels != 2:
        raise ValueError(f"Expected two chroma channels, got shape {chroma.shape}")
    offset_x, offset_y = SITING_OFFSETS[siting]
    weights_x = _downsample_matrix(width, offset_x, filter_name)
    weights_y = _downsample_matrix(height, offset_y, filter_name)
    output = np.empty((height // 2, width // 2, 2), dtype=np.float32)
    for channel in range(2):
        output[..., channel] = weights_y @ chroma[..., channel] @ weights_x.T
    return output


def upsample_chroma(
    low_chroma: np.ndarray,
    output_shape: tuple[int, int],
    siting: str = "center",
    filter_name: str = "bilinear",
) -> np.ndarray:
    if siting not in SITING_OFFSETS:
        raise ValueError(f"Unsupported chroma siting: {siting}")
    if filter_name not in UPSAMPLE_FILTERS:
        raise ValueError(f"Unsupported upsampling filter: {filter_name}")
    output_height, output_width = output_shape
    offset_x, offset_y = SITING_OFFSETS[siting]
    weights_x = _upsample_matrix(
        low_chroma.shape[1], output_width, offset_x, filter_name
    )
    weights_y = _upsample_matrix(
        low_chroma.shape[0], output_height, offset_y, filter_name
    )
    output = np.empty((output_height, output_width, 2), dtype=np.float32)
    for channel in range(2):
        output[..., channel] = weights_y @ low_chroma[..., channel] @ weights_x.T
    return output


def simulate_420(
    target_ycrcb: np.ndarray, spec: DegradationSpec
) -> tuple[np.ndarray, np.ndarray]:
    spec.validate()
    if target_ycrcb.ndim != 3 or target_ycrcb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 YCrCb image, got {target_ycrcb.shape}")
    low = downsample_chroma(
        target_ycrcb[..., 1:3], spec.siting, spec.downsample_filter
    )
    restored = upsample_chroma(
        low, target_ycrcb.shape[:2], spec.siting, spec.upsample_filter
    )
    model_input = np.concatenate((target_ycrcb[..., 0:1], restored), axis=2)
    return model_input.astype(np.float32), low.astype(np.float32)


def deterministic_crop(
    image: np.ndarray,
    record_id: str,
    crop_size: int | None,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    height, width = image.shape[:2]
    if crop_size is None or crop_size == 0:
        even_height = height - height % 2
        even_width = width - width % 2
        return image[:even_height, :even_width], {
            "top": 0,
            "left": 0,
            "height": even_height,
            "width": even_width,
        }
    if crop_size < 8 or crop_size % 2:
        raise ValueError("crop_size must be zero or an even integer of at least 8")
    if height < crop_size or width < crop_size:
        raise ValueError(f"image {width}x{height} is smaller than crop {crop_size}")
    digest = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).digest()
    random_seed = int.from_bytes(digest[:8], "big")
    generator = np.random.default_rng(random_seed)
    top = int(generator.integers(0, height - crop_size + 1))
    left = int(generator.integers(0, width - crop_size + 1))
    return image[top : top + crop_size, left : left + crop_size], {
        "top": top,
        "left": left,
        "height": crop_size,
        "width": crop_size,
    }


def records_for_split(
    records: Iterable[ManifestRecord], split: str
) -> list[ManifestRecord]:
    selected = [record for record in records if record.split == split]
    if not selected:
        raise ValueError(f"Manifest has no records for split {split!r}")
    return selected
