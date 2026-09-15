"""New chroma-only models; original V5/V6/V7 checkpoint formats stay separate."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .models import ConvBlock, ResBlock
from .research_models import NAFStyleBlock
from .research_data import DegradationSpec, SITING_OFFSETS

FILTERS = ("box", "triangle", "gaussian", "lanczos3", "point")


def degradation_features(specs, device, dtype) -> torch.Tensor:
    """Known observation metadata: x/y siting plus a downsampling-filter one-hot."""
    rows = []
    for spec in specs:
        spec.validate()
        rows.append(
            [
                *SITING_OFFSETS[spec.siting],
                *[float(spec.downsample_filter == f) for f in FILTERS],
            ]
        )
    return torch.tensor(rows, device=device, dtype=dtype)


@dataclass(frozen=True)
class PrismArchitecture:
    backbone: str = "residual"
    representation: str = "cartesian"
    width: int = 64
    depth: int = 8
    uncertainty: bool = False
    conditioned: bool = False
    detached_uncertainty: bool = False

    def validate(self):
        if self.backbone not in {"residual", "naf", "unet"}:
            raise ValueError(f"Unknown Prism backbone: {self.backbone}")
        if self.representation not in {"cartesian", "polar"}:
            raise ValueError("representation must be cartesian or polar")
        if self.width < 4 or self.depth < 1:
            raise ValueError("width >= 4 and depth >= 1 are required")
        if self.detached_uncertainty and not self.uncertainty:
            raise ValueError("Detached uncertainty requires uncertainty heads")
        if (
            self.uncertainty
            and self.representation == "cartesian"
            and not self.detached_uncertainty
        ):
            raise ValueError(
                "Cartesian polar-uncertainty heads must use detached_uncertainty"
            )


class UNetFeatures(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.down = nn.ModuleList(
            [
                ConvBlock(width, 2 * width),
                ConvBlock(2 * width, 4 * width),
                ConvBlock(4 * width, 8 * width),
            ]
        )
        self.up = nn.ModuleList(
            [
                ConvBlock(12 * width, 4 * width),
                ConvBlock(6 * width, 2 * width),
                ConvBlock(3 * width, width),
            ]
        )

    def forward(self, inputs):
        skips = [inputs]
        for block in self.down:
            skips.append(block(F.avg_pool2d(skips[-1], 2)))
        output = skips.pop()
        for block, skip in zip(self.up, reversed(skips)):
            output = F.interpolate(
                output, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            output = block(torch.cat((output, skip), dim=1))
        return output


@dataclass
class PrismPrediction:
    image: torch.Tensor
    baseline: torch.Tensor
    amplitude: torch.Tensor
    phase: torch.Tensor
    scale: torch.Tensor | None = None
    kappa: torch.Tensor | None = None

    def confidence(self):
        """Inference-only inputs: no target-dependent confidence weighting."""
        if self.scale is None or self.kappa is None:
            raise ValueError("This Prism model has no uncertainty heads")
        relevance = (self.amplitude / 0.05).clamp(0.0, 1.0)
        return torch.exp(-self.scale / 0.05) * (
            1.0 - relevance + relevance * self.kappa / (self.kappa + 2.0)
        )

    def safe_image(self):
        chroma = self.baseline[:, 1:3] + self.confidence() * (
            self.image[:, 1:3] - self.baseline[:, 1:3]
        )
        return torch.cat((self.baseline[:, :1], chroma), dim=1)


class PrismRefiner(nn.Module):
    def __init__(self, architecture: PrismArchitecture):
        super().__init__()
        architecture.validate()
        self.architecture = architecture
        self.head = nn.Conv2d(3, architecture.width, 3, padding=1)
        if architecture.backbone == "unet":
            self.body = UNetFeatures(architecture.width)
        else:
            block = NAFStyleBlock if architecture.backbone == "naf" else ResBlock
            self.body = nn.Sequential(
                *(block(architecture.width) for _ in range(architecture.depth))
            )
        self.tail = nn.Conv2d(
            architecture.width,
            3 if architecture.representation == "polar" else 2,
            3,
            padding=1,
        )
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)
        if architecture.representation == "polar":
            with torch.no_grad():
                self.tail.bias[1] = 1.0
        self.condition = (
            nn.Linear(7, architecture.width) if architecture.conditioned else None
        )
        self.uncertainty_head = (
            nn.Conv2d(architecture.width, 2, 3, padding=1)
            if architecture.uncertainty
            else None
        )
        if self.uncertainty_head is not None:
            nn.init.zeros_(self.uncertainty_head.weight)
            with torch.no_grad():
                self.uncertainty_head.bias.copy_(
                    torch.tensor([math.log(math.expm1(0.05)), 0.0])
                )

    def predict(self, inputs, specs: list[DegradationSpec] | None = None):
        if inputs.ndim != 4 or inputs.shape[1] != 3 or min(inputs.shape[-2:]) < 8:
            raise ValueError("Prism expects Bx3xHxW float YCrCb with H,W >= 8")
        features = self.head(inputs)
        if self.condition is not None:
            if specs is None or len(specs) != len(inputs):
                raise ValueError(
                    "Conditioned Prism requires one known degradation per image"
                )
            metadata = degradation_features(specs, inputs.device, features.dtype)
            features = features + self.condition(metadata)[:, :, None, None]
        features = self.body(features)
        raw = self.tail(features).float()
        baseline = inputs.float()
        if self.architecture.representation == "polar":
            centered = baseline[:, 1:3] - 0.5
            amplitude0 = torch.linalg.vector_norm(centered, dim=1, keepdim=True)
            phase0 = torch.atan2(centered[:, 1:2], centered[:, 0:1])
            # Smooth nonnegative radius, initialized to A0 (within 5e-6 at gray).
            # No hard clipping of the reconstruction during optimization.
            scaled = (amplitude0 / 0.05).clamp_min(1e-4)
            inverse = scaled + torch.log(-torch.expm1(-scaled))
            amplitude = 0.05 * F.softplus(inverse + raw[:, :1] / 0.05)
            x, y = raw[:, 1:2], raw[:, 2:3]
            small = x.square() + y.square() < 1e-12
            angle = torch.atan2(
                torch.where(small, torch.zeros_like(y), y),
                torch.where(small, torch.ones_like(x), x),
            )
            phase = phase0 + angle
            chroma = 0.5 + torch.cat(
                (amplitude * torch.cos(phase), amplitude * torch.sin(phase)), dim=1
            )
        else:
            chroma = baseline[:, 1:3] + raw
            centered = chroma - 0.5
            # These polar summaries only serve detached auxiliary training/reporting.
            detached = centered.detach()
            amplitude = torch.linalg.vector_norm(detached, dim=1, keepdim=True)
            phase = torch.atan2(detached[:, 1:2], detached[:, 0:1])
        scale = kappa = None
        if self.uncertainty_head is not None:
            aux_features = (
                features.detach()
                if self.architecture.detached_uncertainty
                else features
            )
            auxiliary = self.uncertainty_head(aux_features).float()
            scale = F.softplus(auxiliary[:, :1]) + 1e-5
            kappa = 100.0 * torch.sigmoid(auxiliary[:, 1:2] - math.log(99.0))
        image = torch.cat((baseline[:, :1], chroma), dim=1)
        return PrismPrediction(image, baseline, amplitude, phase, scale, kappa)

    def forward(self, inputs, specs=None):
        return self.predict(inputs, specs).image


class PrismDiscriminator(nn.Module):
    """Patch discriminator conditioned on observed Y/Cr/Cb and candidate Cr/Cb."""

    def __init__(self, width=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(5, width, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(width, 2 * width, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * width, 4 * width, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * width, 1, 3, padding=1),
        )

    def forward(self, observation, candidate):
        return self.layers(torch.cat((observation, candidate[:, 1:3]), dim=1))
