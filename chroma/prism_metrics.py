"""Shared Prism validation metrics with a numerically safe SSIM denominator."""

import torch
import torch.nn.functional as F

from .v7_training import _ycrcb_to_rgb_tensor


def psnr_ssim(reference, candidate):
    reference, candidate = reference.float(), candidate.float()
    mse = (reference - candidate).square().flatten(1).mean(1)
    psnr = (-10.0 * torch.log10(mse.clamp_min(1e-8))).clamp_max(80.0)
    size = min(11, min(reference.shape[-2:]))
    size -= int(size % 2 == 0)
    if size < 3:
        raise ValueError("SSIM requires dimensions >= 3")
    coordinates = (
        torch.arange(size, device=reference.device, dtype=reference.dtype)
        - (size - 1) / 2
    )
    kernel = torch.exp(-coordinates.square() / (2 * (1.5 * size / 11) ** 2))
    kernel /= kernel.sum()
    kernel = torch.outer(kernel, kernel).expand(reference.shape[1], 1, size, size)
    blur = lambda x: F.conv2d(
        F.pad(x, (size // 2,) * 4, mode="reflect"), kernel, groups=reference.shape[1]
    )
    mu_x, mu_y = blur(reference), blur(candidate)
    vx = (blur(reference.square()) - mu_x.square()).clamp_min(0)
    vy = (blur(candidate.square()) - mu_y.square()).clamp_min(0)
    covariance = blur(reference * candidate) - mu_x * mu_y
    bound = (vx * vy).sqrt()
    covariance = torch.maximum(-bound, torch.minimum(bound, covariance))
    numerator = (2 * mu_x * mu_y + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (mu_x.square() + mu_y.square() + 0.01**2) * (vx + vy + 0.03**2)
    ssim = numerator / denominator.clamp_min(torch.finfo(reference.dtype).tiny)
    return psnr, ssim.flatten(1).mean(1)


def quality_batch(target, candidate):
    """Per-image metrics; every model uses the same final [0,1] clipping."""
    outside = ((candidate < 0) | (candidate > 1)).float().flatten(1).mean(1)
    candidate = candidate.clamp(0, 1)
    cp, cs = psnr_ssim(target[:, 1:3], candidate[:, 1:3])
    rp, rs = psnr_ssim(_ycrcb_to_rgb_tensor(target), _ycrcb_to_rgb_tensor(candidate))
    return {
        "chroma_l1": (target[:, 1:3] - candidate[:, 1:3]).abs().flatten(1).mean(1),
        "full_l1": (target - candidate).abs().flatten(1).mean(1),
        "luma_l1": (target[:, :1] - candidate[:, :1]).abs().flatten(1).mean(1),
        "chroma_psnr": cp,
        "chroma_ssim": cs,
        "rgb_psnr": rp,
        "rgb_ssim": rs,
        "out_of_range_fraction": outside,
    }
