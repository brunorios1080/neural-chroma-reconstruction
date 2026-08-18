"""Actual JPEG and optional FFmpeg video-codec round trips."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .research_data import save_rgb


@dataclass(frozen=True)
class CodecResult:
    rgb: np.ndarray
    encoded_bytes: int
    codec: str
    setting_name: str
    setting_value: int
    command: list[str] | None = None


def jpeg_roundtrip(
    rgb: np.ndarray,
    quality: int,
    subsampling: int = 2,
) -> CodecResult:
    """Encode and decode an actual 4:2:0 JPEG through the local Pillow backend."""
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    pixels = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,
        optimize=False,
        progressive=False,
    )
    payload = buffer.getvalue()
    with Image.open(io.BytesIO(payload)) as decoded:
        reconstructed = np.asarray(decoded.convert("RGB"), dtype=np.float32) / 255.0
    return CodecResult(
        rgb=reconstructed,
        encoded_bytes=len(payload),
        codec="jpeg_420",
        setting_name="quality",
        setting_value=quality,
    )


VIDEO_ENCODERS = {
    "h264": ("libx264", ["-preset", "medium"]),
    "hevc": ("libx265", ["-preset", "medium"]),
    "av1": ("libaom-av1", ["-cpu-used", "4"]),
}


def _resolve_ffmpeg(executable: str = "ffmpeg") -> str | None:
    requested = executable
    if executable in {"auto", "ffmpeg"}:
        requested = os.environ.get("CHROMA_FFMPEG", "ffmpeg")
    path = shutil.which(requested)
    if path is None:
        candidate = Path(requested).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            path = str(candidate.resolve())
    return path


@lru_cache(maxsize=None)
def ffmpeg_capabilities(executable: str = "ffmpeg") -> dict[str, object]:
    path = _resolve_ffmpeg(executable)
    if path is None:
        return {"available": False, "path": None, "encoders": []}
    process = subprocess.run(
        [path, "-hide_banner", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
    )
    text = process.stdout + process.stderr
    encoders = [encoder for encoder, _ in VIDEO_ENCODERS.values() if encoder in text]
    return {
        "available": process.returncode == 0,
        "path": path,
        "encoders": encoders,
        "returncode": process.returncode,
    }


def video_roundtrip(
    rgb: np.ndarray,
    codec: str,
    quality: int,
    executable: str = "ffmpeg",
) -> CodecResult:
    """Encode one lossless source frame as yuv420p and decode it back to RGB."""
    if codec not in VIDEO_ENCODERS:
        raise ValueError(f"Unsupported video codec {codec!r}; choose {sorted(VIDEO_ENCODERS)}")
    capabilities = ffmpeg_capabilities(executable)
    encoder, encoder_options = VIDEO_ENCODERS[codec]
    if not capabilities["available"]:
        raise RuntimeError("FFmpeg is not installed locally")
    if encoder not in capabilities["encoders"]:
        raise RuntimeError(f"Local FFmpeg does not provide encoder {encoder}")
    ffmpeg = str(capabilities["path"])
    with tempfile.TemporaryDirectory(prefix="chroma_codec_") as temporary:
        directory = Path(temporary)
        source = directory / "source.png"
        encoded = directory / "encoded.mkv"
        decoded = directory / "decoded.png"
        save_rgb(source, rgb)
        quality_options = (
            ["-crf", str(quality), "-b:v", "0"]
            if codec == "av1"
            else ["-crf", str(quality)]
        )
        codec_options = [*encoder_options, *quality_options]
        encode_command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-threads",
            "1",
            "-c:v",
            encoder,
            "-pix_fmt",
            "yuv420p",
            *codec_options,
            str(encoded),
        ]
        subprocess.run(encode_command, check=True, capture_output=True)
        decode_command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(encoded),
            "-frames:v",
            "1",
            str(decoded),
        ]
        subprocess.run(decode_command, check=True, capture_output=True)
        with Image.open(decoded) as image:
            reconstructed = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return CodecResult(
            rgb=reconstructed,
            encoded_bytes=encoded.stat().st_size,
            codec=codec,
            setting_name="crf",
            setting_value=quality,
            command=[
                "ffmpeg",
                "-i",
                "<lossless-source>",
                "-frames:v",
                "1",
                "-threads",
                "1",
                "-c:v",
                encoder,
                "-pix_fmt",
                "yuv420p",
                *codec_options,
                "<encoded.mkv>",
            ],
        )
