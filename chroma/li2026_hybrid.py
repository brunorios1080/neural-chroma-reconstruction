"""Paper-inspired color reconstruction followed by a gated V6/Prism residual.

The analytic implementation is independent, not author code. Neural inference
receives only the decoded baseline and its known sampling layout. It never sees
the original RGB target or untransmitted chroma.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .li2026 import reconstruct
from .models import ChromaRefiner
from .prism_models import PrismArchitecture, PrismRefiner
from .research_data import DegradationSpec

FORMAT = 'li2026-hybrid-v1'


@dataclass(frozen=True)
class HybridProtocol:
    transform: str = 'scaled_decode_first'
    sampling: str = 'cosited_point'
    interpolation: str = 'bicubic'
    matrix: str = 'paper_rounded'

    def validate(self):
        if self.transform not in {'conventional', 'scaled_matrix', 'scaled_decode_first'}:
            raise ValueError('Unsupported hybrid transform')
        if self.sampling not in {'cosited_point', 'center_box'}:
            raise ValueError('Unsupported hybrid sampling')
        if self.interpolation not in {'bilinear', 'bicubic'} or self.matrix != 'paper_rounded':
            raise ValueError('Use the audited paper matrix and bilinear/bicubic interpolation')
        if self.transform == 'scaled_decode_first' and self.sampling != 'cosited_point':
            raise ValueError('Retained-site lifting requires point sampling')

    def spec(self):
        self.validate()
        return DegradationSpec(
            name=self.sampling,
            siting='cosited' if self.sampling == 'cosited_point' else 'center',
            downsample_filter='point' if self.sampling == 'cosited_point' else 'box',
            upsample_filter=self.interpolation,
        )


def prepare_baseline(rgb, protocol):
    """Simulate encoding/decoding; return the baseline plus its retained-site mask."""
    protocol.validate()
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected HxWx3 uint8 RGB source')
    baseline, _, _ = reconstruct(rgb, protocol.sampling, protocol.interpolation,
                                 protocol.matrix, protocol.transform)
    keep = np.zeros(rgb.shape[:2], dtype=bool)
    if protocol.sampling == 'cosited_point':
        keep[::2, ::2] = True
    return baseline.astype(np.float32) / 255., keep


def rgb_to_model_input(rgb, neutral):
    """Float adapter; do not introduce another 8-bit YCrCb quantization step."""
    red, green, blue = rgb[:, :1], rgb[:, 1:2], rgb[:, 2:3]
    luma = .299 * red + .587 * green + .114 * blue
    return torch.cat((luma, neutral + .713 * (red - luma),
                      neutral + .564 * (blue - luma)), dim=1)


class HybridRefiner(nn.Module):
    """Preserve the analytic baseline, then learn a chroma-only RGB correction.

    A zero gain gives bit-for-bit identity on the floating baseline. Retained
    reconstructions are protected even after training; these are exact original
    RGB at retained sites for the lifting variant. Matrix-only retained RGB is
    quantized, and the same projection preserves that baseline reconstruction.
    """
    def __init__(self, backbone, description, protocol, flat_guard_radius=0):
        super().__init__()
        protocol.validate()
        self.backbone, self.description, self.protocol = backbone, description, protocol
        if description['kind'] not in {'v6', 'prism'}:
            raise ValueError('Hybrid supports V6 and Prism backbones')
        self.neutral = 128 / 255 if description['kind'] == 'v6' else .5
        if not isinstance(flat_guard_radius, int) or not 0 <= flat_guard_radius <= 8:
            raise ValueError('flat_guard_radius must be an integer from 0 to 8')
        self.flat_guard_radius = flat_guard_radius
        self.correction_gain = nn.Parameter(torch.zeros(()))
        # Auxiliary uncertainty heads are not supervised by this RGB-MSE pilot.
        auxiliary = getattr(backbone, 'uncertainty_head', None)
        if auxiliary is not None:
            auxiliary.requires_grad_(False)

    def forward(self, baseline, keep):
        if baseline.ndim != 4 or baseline.shape[1] != 3 or min(baseline.shape[-2:]) < 8:
            raise ValueError('Expected Bx3xHxW RGB baseline with H,W >= 8')
        if keep.shape != baseline[:, :1].shape or keep.dtype != torch.bool:
            raise ValueError('Retained-site mask must be Bx1xHxW boolean')
        inputs = rgb_to_model_input(baseline, self.neutral)
        if self.description['kind'] == 'prism':
            predicted = self.backbone(inputs, [self.protocol.spec()] * len(inputs))
        else:
            predicted = self.backbone(inputs)
        difference = predicted[:, 1:3] - inputs[:, 1:3]
        # Exact inverse of the adapter's chroma mapping for a zero-luma residual.
        dr = difference[:, :1] / .713
        db = difference[:, 1:2] / .564
        dg = -(.299 * dr + .114 * db) / .587
        correction = torch.cat((dr, dg, db), dim=1)
        protected = keep
        if self.flat_guard_radius:
            # RGB channel differences encode chromaticity independently of
            # brightness. This also protects gray edges with varying luminance.
            chromatic = torch.cat((baseline[:, :1]-baseline[:, 1:2],
                                   baseline[:, 2:3]-baseline[:, 1:2]), dim=1)
            radius = self.flat_guard_radius
            maximum = F.max_pool2d(chromatic, 2*radius+1, stride=1, padding=radius)
            minimum = -F.max_pool2d(-chromatic, 2*radius+1, stride=1, padding=radius)
            # Tolerance only absorbs float32 cancellation (well below 1/255).
            flat = (maximum-minimum).amax(dim=1, keepdim=True) <= 1e-6
            protected = protected | flat
        correction = correction.masked_fill(protected, 0.)
        return baseline + self.correction_gain.tanh() * correction

    def metadata(self):
        return {'backbone': self.description, 'protocol': asdict(self.protocol),
                'flat_guard_radius': self.flat_guard_radius}


def build_hybrid(metadata):
    description = metadata['backbone']
    if description['kind'] == 'v6':
        backbone = ChromaRefiner()
    elif description['kind'] == 'prism':
        backbone = PrismRefiner(PrismArchitecture(**description['architecture']))
    else:
        raise ValueError('Unsupported hybrid backbone')
    return HybridRefiner(backbone, description, HybridProtocol(**metadata['protocol']),
                         metadata.get('flat_guard_radius', 0))


def initialize_hybrid(entry, protocol, flat_guard_radius=0):
    """Use a verified frozen checkpoint; original checkpoint files are read-only."""
    from .model_comparison import load_candidate
    from .research_data import sha256_file
    if entry['kind'] != 'prism' and not (entry['kind'] == 'legacy' and entry.get('version') == 'v6'):
        raise ValueError('Choose a V6 or Prism checkpoint for hybrid fine-tuning')
    if sha256_file(entry['weights']) != entry['sha256']:
        raise ValueError('Initialization checkpoint checksum mismatch')
    backbone, _ = load_candidate(entry, 'cpu')
    description = {'kind': 'v6'} if entry['kind'] == 'legacy' else {
        'kind': 'prism', 'architecture': asdict(backbone.architecture)}
    return HybridRefiner(backbone, description, protocol, flat_guard_radius)


def load_hybrid(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != FORMAT:
        raise ValueError('Expected a Li-2026 hybrid checkpoint')
    model = build_hybrid(payload['metadata'])
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device), payload


class HybridDataset(Dataset):
    """Deterministic COCO crops, from a directory or a lazily opened ZIP archive."""
    def __init__(self, records, source, crop_size, protocol, seed, training=False):
        if not records or crop_size < 8 or crop_size % 2:
            raise ValueError('Use nonempty records and even crop_size >= 8')
        protocol.validate()
        self.records, self.source = list(records), Path(source)
        self.crop_size, self.protocol, self.seed, self.training = crop_size, protocol, seed, training
        self.archive = None

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        return {**self.__dict__, 'archive': None}

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        record = self.records[index]
        if self.source.is_dir():
            with Image.open(self.source / record.relative_path) as image:
                rgb = np.array(image.convert('RGB'))
        else:
            if self.archive is None:
                self.archive = zipfile.ZipFile(self.source)
            with self.archive.open(record.relative_path) as source, Image.open(source) as image:
                rgb = np.array(image.convert('RGB'))
        height, width = rgb.shape[:2]
        if min(height, width) < self.crop_size:
            raise ValueError('Source smaller than crop: ' + record.id)
        digest = hashlib.sha256(f'{self.seed}:{epoch}:{record.id}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
        if self.training:
            top, left = int(rng.integers(height-self.crop_size+1)), int(rng.integers(width-self.crop_size+1))
        else:
            top, left = (height-self.crop_size)//2, (width-self.crop_size)//2
        rgb = rgb[top:top+self.crop_size, left:left+self.crop_size]
        if self.training:
            rgb = np.rot90(rgb, int(rng.integers(4)))
            if rng.integers(2):
                rgb = np.flip(rgb, axis=1)
        rgb = np.ascontiguousarray(rgb)
        baseline, keep = prepare_baseline(rgb, self.protocol)
        tensor = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)
        return tensor(baseline), tensor(rgb.astype(np.float32)/255), torch.from_numpy(keep[None]), record.id


def rounded_rgb(value):
    return torch.floor(value.clamp(0, 1).double() * 255 + .5)


@torch.no_grad()
def validate_hybrid(model, loader, device):
    model.eval()
    rows = []
    for baseline, target, keep, identifiers in loader:
        baseline, target, keep = baseline.to(device), target.to(device), keep.to(device)
        prediction = model(baseline, keep)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite hybrid reconstruction')
        if not torch.equal(prediction.masked_select(keep), baseline.masked_select(keep)):
            raise ValueError('Hybrid changed a protected retained sample')
        original, analytic, learned = rounded_rgb(target), rounded_rgb(baseline), rounded_rgb(prediction)
        for i, identifier in enumerate(identifiers):
            metrics = {}
            for name, candidate in [('baseline', analytic), ('hybrid', learned)]:
                mse = float((candidate[i]-original[i]).square().mean())
                metrics[name+'_mse_rgb_255'] = mse
                metrics[name+'_cpsnr_rgb'] = float(10 * np.log10(255**2 / max(mse, 1e-12)))
                metrics[name+'_mae_rgb_255'] = float((candidate[i]-original[i]).abs().mean())
            rows.append({'id': identifier, **metrics})
    if not rows:
        raise ValueError('No validation images')
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != 'id'}
    summary.update(images=len(rows), correction_gain=float(model.correction_gain.tanh()),
                   mean_cpsnr_gain_db=summary['hybrid_cpsnr_rgb']-summary['baseline_cpsnr_rgb'],
                   wins=sum(r['hybrid_cpsnr_rgb'] > r['baseline_cpsnr_rgb'] for r in rows),
                   baseline_perfect_images=sum(r['baseline_mse_rgb_255'] == 0 for r in rows),
                   hybrid_perfect_images=sum(r['hybrid_mse_rgb_255'] == 0 for r in rows))
    return summary, rows
