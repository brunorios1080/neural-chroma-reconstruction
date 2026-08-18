"""Configurable learned models and ablation losses for the research protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class AblationConfig:
    name: str
    architecture: str = "residual"
    luma_guidance: bool = True
    global_residual: bool = True
    block_residual: bool = True
    depth: int = 8
    width: int = 64
    loss: str = "l1"
    edge_weight: float = 0.2

    def validate(self) -> None:
        if self.architecture not in {"residual", "srcnn", "naf_style"}:
            raise ValueError(
                f"Unsupported research architecture: {self.architecture}"
            )
        if self.depth < 1:
            raise ValueError("Ablation depth must be positive")
        if self.width < 4:
            raise ValueError("Ablation width must be at least four")
        if self.loss not in {"l1", "mse", "charbonnier", "l1_edge"}:
            raise ValueError(f"Unsupported ablation loss: {self.loss}")
        if self.edge_weight < 0.0:
            raise ValueError("edge_weight cannot be negative")


class ConfigurableResBlock(nn.Module):
    def __init__(self, channels: int, residual: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.residual = residual

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.conv2(self.relu(self.conv1(inputs)))
        return inputs + output if self.residual else output


class AblationChromaRefiner(nn.Module):
    """V6-compatible family with explicit luma, residual, depth, and width knobs."""

    def __init__(self, config: AblationConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        input_channels = 3 if config.luma_guidance else 2
        self.head = nn.Conv2d(input_channels, config.width, 3, padding=1)
        self.body = nn.Sequential(
            *(
                ConfigurableResBlock(config.width, config.block_residual)
                for _ in range(config.depth)
            )
        )
        self.tail = nn.Conv2d(config.width, 2, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError(f"Expected Bx3xHxW input, got {tuple(inputs.shape)}")
        network_input = inputs if self.config.luma_guidance else inputs[:, 1:3]
        prediction = self.tail(self.body(self.head(network_input)))
        if self.config.global_residual:
            chroma = inputs[:, 1:3] + prediction
        else:
            chroma = torch.sigmoid(prediction)
        return torch.cat((inputs[:, 0:1], chroma), dim=1)


class SrcnnChromaRefiner(nn.Module):
    """Three-layer SRCNN adapted to guided, same-resolution chroma recovery."""

    def __init__(self, config: AblationConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        input_channels = 3 if config.luma_guidance else 2
        bottleneck = max(4, config.width // 2)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, config.width, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.width, bottleneck, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, 2, 5, padding=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError(f"Expected Bx3xHxW input, got {tuple(inputs.shape)}")
        network_input = inputs if self.config.luma_guidance else inputs[:, 1:3]
        prediction = self.features(network_input)
        chroma = (
            inputs[:, 1:3] + prediction
            if self.config.global_residual
            else torch.sigmoid(prediction)
        )
        return torch.cat((inputs[:, 0:1], chroma), dim=1)


class LayerNorm2d(nn.Module):
    """Per-pixel channel normalization used by NAF-style restoration blocks."""

    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=1, keepdim=True)
        variance = (inputs - mean).square().mean(dim=1, keepdim=True)
        normalized = (inputs - mean) * torch.rsqrt(variance + self.epsilon)
        return normalized * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first, second = inputs.chunk(2, dim=1)
        return first * second


class NAFStyleBlock(nn.Module):
    """Nonlinear-activation-free restoration block adapted from NAFNet."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        expanded = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, expanded, 1)
        self.depthwise = nn.Conv2d(
            expanded, expanded, 3, padding=1, groups=expanded
        )
        self.gate1 = SimpleGate()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
        )
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, expanded, 1)
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.depthwise(self.conv1(self.norm1(inputs)))
        features = self.gate1(features)
        features = features * self.channel_attention(features)
        first_residual = inputs + self.beta * self.conv2(features)
        features = self.gate2(self.conv3(self.norm2(first_residual)))
        return first_residual + self.gamma * self.conv4(features)


class NAFStyleChromaRefiner(nn.Module):
    """A shallow NAF-style same-resolution baseline for chroma restoration."""

    def __init__(self, config: AblationConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        input_channels = 3 if config.luma_guidance else 2
        self.head = nn.Conv2d(input_channels, config.width, 3, padding=1)
        self.body = nn.Sequential(
            *(NAFStyleBlock(config.width) for _ in range(config.depth))
        )
        self.tail = nn.Conv2d(config.width, 2, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError(f"Expected Bx3xHxW input, got {tuple(inputs.shape)}")
        network_input = inputs if self.config.luma_guidance else inputs[:, 1:3]
        prediction = self.tail(self.body(self.head(network_input)))
        chroma = (
            inputs[:, 1:3] + prediction
            if self.config.global_residual
            else torch.sigmoid(prediction)
        )
        return torch.cat((inputs[:, 0:1], chroma), dim=1)


def _spatial_gradients(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    horizontal = values[..., :, 1:] - values[..., :, :-1]
    vertical = values[..., 1:, :] - values[..., :-1, :]
    return horizontal, vertical


def chroma_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: AblationConfig,
) -> torch.Tensor:
    predicted_chroma = prediction[:, 1:3]
    target_chroma = target[:, 1:3]
    if config.loss == "l1":
        return F.l1_loss(predicted_chroma, target_chroma)
    if config.loss == "mse":
        return F.mse_loss(predicted_chroma, target_chroma)
    if config.loss == "charbonnier":
        difference = predicted_chroma - target_chroma
        return torch.mean(torch.sqrt(difference * difference + 1e-6))
    if config.loss == "l1_edge":
        reconstruction = F.l1_loss(predicted_chroma, target_chroma)
        predicted_dx, predicted_dy = _spatial_gradients(predicted_chroma)
        target_dx, target_dy = _spatial_gradients(target_chroma)
        edges = 0.5 * (
            F.l1_loss(predicted_dx, target_dx)
            + F.l1_loss(predicted_dy, target_dy)
        )
        return reconstruction + config.edge_weight * edges
    raise ValueError(f"Unsupported loss: {config.loss}")


def build_ablation_model(config: AblationConfig | Mapping[str, Any]) -> nn.Module:
    if not isinstance(config, AblationConfig):
        config = AblationConfig(**dict(config))
    builders: dict[str, type[nn.Module]] = {
        "residual": AblationChromaRefiner,
        "srcnn": SrcnnChromaRefiner,
        "naf_style": NAFStyleChromaRefiner,
    }
    return builders[config.architecture](config)


def save_ablation_checkpoint(
    path: str | Path,
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "format_version": 2,
        "model_family": "ablation_chroma_refiner",
        "config": asdict(model.config),  # type: ignore[attr-defined]
        "model": model.state_dict(),
        "epoch": int(epoch),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(dict(extra))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_ablation_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_family") != "ablation_chroma_refiner":
        raise ValueError(f"Not a research ablation checkpoint: {path}")
    model = build_ablation_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_bytes(model: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def convolution_flops(
    model: nn.Module, input_shape: tuple[int, int, int, int], device: torch.device
) -> int:
    """Count multiply and add as two FLOPs for Conv2d/ConvTranspose2d layers."""
    total = 0
    handles = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
        nonlocal total
        del inputs
        if isinstance(module, nn.Conv2d):
            kernel_ops = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * module.in_channels
                // module.groups
            )
        elif isinstance(module, nn.ConvTranspose2d):
            kernel_ops = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * module.in_channels
                // module.groups
            )
        else:
            return
        total += int(output.numel() * kernel_ops * 2)

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(layer.register_forward_hook(hook))
    training = model.training
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(input_shape, device=device))
    for handle in handles:
        handle.remove()
    model.train(training)
    return total
