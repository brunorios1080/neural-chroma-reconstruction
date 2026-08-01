"""Model definitions shared by training, evaluation, and inference."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UNetGenerator(nn.Module):
    """V5 U-Net generator.

    The attribute names intentionally match the original research script so that
    the committed V5 checkpoint remains loadable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.e1 = ConvBlock(3, 32)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = ConvBlock(32, 64)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = ConvBlock(64, 128)
        self.p3 = nn.MaxPool2d(2)
        self.e4 = ConvBlock(128, 256)

        self.u3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d3 = ConvBlock(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d2 = ConvBlock(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(inputs)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        bottleneck = self.e4(self.p3(e3))

        d3 = self.d3(torch.cat([self.u3(bottleneck), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))


class Discriminator(nn.Module):
    """V5 PatchGAN discriminator."""

    def __init__(self) -> None:
        super().__init__()
        channels = 32
        self.model = nn.Sequential(
            nn.Conv2d(3, channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, channels * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 4, channels * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 8, 1, 4, 1, 0),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.relu(self.conv1(inputs)))
        return inputs + residual


class ChromaRefiner(nn.Module):
    """V6 residual model that leaves Y unchanged and refines Cr/Cb."""

    def __init__(
        self, in_channels: int = 3, features: int = 64, num_blocks: int = 8
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, features, 3, padding=1)
        self.body = nn.Sequential(*(ResBlock(features) for _ in range(num_blocks)))
        self.tail = nn.Conv2d(features, 2, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.body(self.head(inputs))
        refined_chroma = inputs[:, 1:3] + self.tail(features)
        return torch.cat([inputs[:, 0:1], refined_chroma], dim=1)


def normalize_version(version: str) -> str:
    normalized = version.lower().removeprefix("model").strip()
    if normalized not in {"v5", "v6"}:
        raise ValueError(
            f"Unsupported model version: {version!r}; expected 'v5' or 'v6'"
        )
    return normalized


def build_model(version: str) -> nn.Module:
    version = normalize_version(version)
    return UNetGenerator() if version == "v5" else ChromaRefiner()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
