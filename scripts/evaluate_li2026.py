#!/usr/bin/env python3
"""Reproduce published baselines, then compare frozen checkpoints on shared data."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chroma.li2026 import (MATRICES, cpsnr_mae, iround, lift_forward, lift_inverse,
    lifting_factors, model_input, model_output, reconstruct, scaled_transform)
from chroma.model_comparison import load_candidate, predict_candidate
from chroma.research_data import DegradationSpec, sha256_file
from chroma.research_metrics import srgb_to_lab, delta_e_ciede2000_lab

PUBLISHED = {
    'uhd240': {'bilinear': {'cpsnr_rgb': 50.49, 'mae_rgb_255': .48},
               'bicubic': {'cpsnr_rgb': 50.69, 'mae_rgb_255': .46},
               'scaled_bilinear': {'cpsnr_rgb': 54.40, 'mae_rgb_255': .19},
               'scaled_bicubic': {'cpsnr_rgb': 54.82, 'mae_rgb_255': .18}},
    'kodak': {'bilinear': {'cpsnr_rgb': 47.33, 'mae_rgb_255': .60},
              'bicubic': {'cpsnr_rgb': 46.28, 'mae_rgb_255': .69},
              'scaled_bilinear': {'cpsnr_rgb': 48.31, 'mae_rgb_255': .43},
              'scaled_bicubic': {'cpsnr_rgb': 47.42, 'mae_rgb_255': .50}}}
TRAINED = {'v5', 'v5_1', 'v6', 'v7', 'prism_residual', 'prism_polar',
           'prism_polar_prob', 'prism_cartesian_prob'}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def records(root, dataset):
    if dataset == 'kodak':
        paths = sorted((root / 'kodak').glob('kodim*.png'))
        if len(paths) != 24:
            raise ValueError(f'Kodak must contain 24 images, found {len(paths)}')
        return [{'id': path.name, 'path': str(path), 'dataset': 'kodak'} for path in paths]
    if dataset not in ('uhd240', 'uhd240_4k', 'uhd240_6k', 'uhd240_8k'):
        raise ValueError(dataset)
    result = []
    for resolution in ('4K', '6K', '8K'):
        if dataset != 'uhd240' and not dataset.endswith(resolution.lower()):
            continue
        path = root / 'uhd240' / f'{resolution}_80.zip'
        with zipfile.ZipFile(path) as archive:
            members = sorted(item for item in archive.namelist()
                if Path(item).suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                and '__MACOSX' not in item and not Path(item).name.startswith('.'))
        if len(members) != 80:
            raise ValueError(f'{path} must contain 80 images, found {len(members)}')
        result.extend({'id': resolution + '/' + item, 'archive': str(path), 'member': item,
                       'dataset': 'uhd240', 'resolution': resolution} for item in members)
    return result


def read_image(record):
    if 'archive' in record:
        with zipfile.ZipFile(record['archive']) as archive:
            with archive.open(record['member']) as source:
                with Image.open(source) as image:
                    value = np.array(image.convert('RGB'))
    else:
        with Image.open(record['path']) as image:
            value = np.array(image.convert('RGB'))
    return value


def infer(model, kind, inputs, sampling, device, tile=512, halo=64):
    h, w = inputs.shape[:2]
    # The historical U-Net requires multiples of eight; apply only right/bottom
    # edge padding and remove it after inference. Tile origins remain aligned.
    pad_h, pad_w = (-h) % 8, (-w) % 8
    padded = np.pad(inputs, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    ph, pw = padded.shape[:2]
    output = np.empty_like(padded)
    spec = DegradationSpec(name=sampling, siting='cosited' if sampling == 'cosited_point' else 'center',
                           downsample_filter='point' if sampling == 'cosited_point' else 'box')
    step = tile or max(ph, pw)
    with torch.inference_mode():
        for top in range(0, ph, step):
            for left in range(0, pw, step):
                bottom, right = min(top + step, ph), min(left + step, pw)
                y0, x0 = max(0, top - halo), max(0, left - halo)
                y1, x1 = min(ph, bottom + halo), min(pw, right + halo)
                tensor = torch.from_numpy(np.ascontiguousarray(padded[y0:y1, x0:x1].transpose(2, 0, 1))).unsqueeze(0).to(device)
                prediction = predict_candidate(model, kind, tensor, [spec])[0]
                core = prediction[:, top-y0:bottom-y0, left-x0:right-x0]
                output[top:bottom, left:right] = core.permute(1, 2, 0).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite model output')
    return np.clip(output[:h, :w], 0, 1)


def quality_cpu(reference, candidate, device, extras=True):
    result = cpsnr_mae(reference, candidate)
    if not extras:
        return result
    from piq import fsim
    coefficients = np.array([.299, .587, .114])
    height, width = reference.shape[:2]
    de_sum, ssim_sum, ssim_count = 0., 0., 0
    # Perceptual metrics use every pixel; strip processing only bounds RAM.
    for top in range(0, height, 256):
        bottom = min(top + 256, height)
        a, b = reference[top:bottom] / 255., candidate[top:bottom] / 255.
        de_sum += float(delta_e_ciede2000_lab(srgb_to_lab(a), srgb_to_lab(b)).sum())
        y0, y1 = max(0, top - 5), min(height, bottom + 5)
        x = reference[y0:y1].astype(np.float64) @ coefficients
        y = candidate[y0:y1].astype(np.float64) @ coefficients
        blur = lambda v: cv2.GaussianBlur(v, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT_101)
        mx, my = blur(x), blur(y)
        vx, vy, covariance = blur(x*x) - mx*mx, blur(y*y) - my*my, blur(x*y) - mx*my
        score = ((2*mx*my + 2.55**2) * (2*covariance + 7.65**2) /
                 ((mx*mx + my*my + 2.55**2) * (vx + vy + 7.65**2)))
        lo, hi = max(top, 5)-y0, min(bottom, height-5)-y0
        if hi > lo:
            valid = score[lo:hi, 5:-5]
            ssim_sum += float(valid.sum())
            ssim_count += valid.size
    with torch.inference_mode():
        a = torch.from_numpy(np.ascontiguousarray(reference.transpose(2, 0, 1))).unsqueeze(0).float().to(device) / 255
        b = torch.from_numpy(np.ascontiguousarray(candidate.transpose(2, 0, 1))).unsqueeze(0).float().to(device) / 255
        score = float(fsim(a, b, data_range=1., chromatic=False).item())
    result.update(ciede2000=de_sum/(height*width), ssim_luma=ssim_sum/ssim_count, fsim_luma=score)
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError(f'Nonfinite quality metric: {result}')
    return result


def quality(reference, candidate, device, extras=True):
    if not device.startswith('cuda') or not extras:
        return quality_cpu(reference, candidate, device, extras)
    from chroma.li2026_metrics import perceptual
    from piq import fsim
    result = cpsnr_mae(reference, candidate)
    result.update(perceptual(reference, candidate, device))
    with torch.inference_mode():
        a = torch.as_tensor(np.ascontiguousarray(reference.transpose(2, 0, 1)), device=device, dtype=torch.float32).unsqueeze(0) / 255
        b = torch.as_tensor(np.ascontiguousarray(candidate.transpose(2, 0, 1)), device=device, dtype=torch.float32).unsqueeze(0) / 255
        result['fsim_luma'] = float(fsim(a, b, data_range=1., chromatic=False).item())
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError(f'Nonfinite quality metric: {result}')
    return result


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['sampling'], row['method'])].append(row)
    result = []
    for (dataset, sampling, method), values in sorted(groups.items()):
        metrics = {key: float(np.mean([row['metrics'][key] for row in values]))
                   for key in values[0]['metrics']}
        result.append({'dataset': dataset, 'sampling': sampling, 'method': method,
                       'images': len(values), 'metrics': metrics})
    return result


def calibration(args):
    args.output.mkdir(parents=True, exist_ok=True)
    # Reproduce Table 1 over all 256^3 RGB values without allocating the cube.
    counts = {}
    for name, a0 in MATRICES.items():
        scale, a = scaled_transform(a0)
        factors = lifting_factors(a)
        totals = {'conventional': 0, 'scaled_matrix': 0, 'scaled_lifting': 0}
        gb = np.indices((256, 256)).reshape(2, -1).T
        for red in range(256):
            rgb = np.column_stack((np.full(len(gb), red), gb)).astype(np.float64)
            for key, matrix in [('conventional', a0), ('scaled_matrix', a)]:
                reconstructed = iround(iround(rgb @ matrix.T) @ np.linalg.inv(matrix).T)
                totals[key] += int(np.count_nonzero(reconstructed == rgb))
            totals['scaled_lifting'] += int(np.count_nonzero(lift_inverse(lift_forward(rgb, factors), factors) == rgb))
        literal = lifting_factors(a, literal_equation27=True)
        counts[name] = {'scale': scale.tolist(), 'matrix': a.tolist(), 'counts': totals,
                        'literal_equation27_factorization_max_error': float(np.abs(literal[0] @ literal[1] @ literal[2] - a).max()),
                        'paper_counts': [30378628, 45095625, 50331648]}
        write_json(args.output / 'cube_checks.json', counts)
        print(json.dumps({'cube_check': name, **counts[name]}), flush=True)
    rows = []
    for record in records(args.data, 'kodak'):
        rgb = read_image(record)
        for matrix in MATRICES:
            for sampling in ('cosited_point', 'center_box'):
                for method in ('bilinear', 'bicubic'):
                    transforms = ('conventional', 'scaled_matrix', 'scaled_hybrid', 'scaled_lifting', 'scaled_decode_first') if sampling == 'cosited_point' else ('conventional', 'scaled_matrix')
                    for transform in transforms:
                        prediction, _, _ = reconstruct(rgb, sampling, method, matrix, transform)
                        rows.append({'dataset': 'kodak', 'id': record['id'], 'sampling': sampling,
                                     'method': f'{matrix}/{transform}/{method}',
                                     'metrics': cpsnr_mae(rgb, prediction)})
        print(json.dumps({'calibrated': record['id']}), flush=True)
        write_json(args.output / 'calibration.json', {'status': 'running', 'results': summarize(rows)})
    write_json(args.output / 'calibration_rows.json', rows)
    write_json(args.output / 'calibration.json', {'status': 'complete', 'results': summarize(rows),
        'published': PUBLISHED, 'warning': 'Matching a rounded aggregate alone does not establish exact reproduction.'})


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.campaign.read_text())
    entries = [row for row in source['models'] if row['name'] in TRAINED]
    if args.models:
        entries = [row for row in entries if row['name'] in args.models.split(',')]
    loaded = []
    for entry in entries:
        if sha256_file(entry['weights']) != entry['sha256']:
            raise ValueError('Checkpoint checksum mismatch: ' + entry['name'])
        model, metadata = load_candidate(entry, args.device)
        loaded.append((entry, model))
    manifest = [record for dataset in args.datasets.split(',') for record in records(args.data, dataset)]
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid shard index/count')
    manifest = manifest[args.shard_index::args.shard_count]
    if args.limit:
        manifest = manifest[:args.limit]
    configuration = {'source_campaign_sha256': sha256_file(args.campaign), 'models': entries,
        'implementation_sha256': {str(path.relative_to(Path(__file__).resolve().parents[1])): sha256_file(path)
            for path in sorted((Path(__file__).resolve().parents[1] / 'chroma').glob('*.py'))},
        'runner_sha256': sha256_file(__file__),
        'manifest': manifest, 'matrix': args.matrix, 'sampling': args.sampling.split(','),
        'shard_index': args.shard_index, 'shard_count': args.shard_count,
        'tile': args.tile, 'halo': 64, 'precision': 'model float32 TF32 disabled; transforms/PSNR float64; CIEDE2000/SSIM CUDA float64',
        'metric_definitions': {'cpsnr_rgb': 'Eq. 42–43, all RGB pixels, average per-image dB',
           'mae_rgb_255': 'Eq. 47, all RGB pixels, 0–255 units',
           'ssim_luma': '11x11 Gaussian sigma 1.5, valid windows, BT.601 luma, L=255',
           'ciede2000': 'all RGB pixels converted sRGB D65 Lab; kL=kC=kH=1',
           'fsim_luma': 'PIQ 0.8.0 fsim(chromatic=False), default scales and automatic pooling'},
        'rounding': 'ties away from zero; final RGB clipped [0,255]; model YCrCb clipped [0,1]',
        'model_neutral_chroma': '128/255 for legacy OpenCV-trained models; 0.5 for Prism/V7',
        'author_code': False, 'exact_published_protocol_verified': False,
        'equation27': 'Matrix-consistent -M31 used; online equation prints +M31. Literal formula audited separately.',
        'missing_details': ['sampling filter and siting', 'edge handling',
                            'lifting/matrix transition', 'metric implementation details']}
    path = args.output / 'configuration.json'
    if path.exists() and json.loads(path.read_text()) != configuration:
        raise ValueError('Resume configuration differs')
    write_json(path, configuration)
    previous = args.output / 'rows.jsonl'
    rows = [json.loads(line) for line in previous.read_text().splitlines()] if previous.exists() else []
    done = {(row['id'], row['sampling'], row['method']) for row in rows}
    started = time.monotonic()
    for record in manifest:
        rgb = read_image(record)
        checksum = hashlib.sha256(rgb.tobytes()).hexdigest()
        for sampling in args.sampling.split(','):
            observation = None
            for interpolation in ('bilinear', 'bicubic'):
                transforms = ('conventional', 'scaled_matrix', 'scaled_hybrid', 'scaled_lifting', 'scaled_decode_first') if sampling == 'cosited_point' else ('conventional', 'scaled_matrix')
                for transform in transforms:
                    name = transform + '/' + interpolation
                    need = (record['id'], sampling, name) not in done
                    if need or (transform == 'conventional' and interpolation == 'bilinear'):
                        prediction, observed, _ = reconstruct(rgb, sampling, interpolation, args.matrix, transform)
                        if transform == 'conventional' and interpolation == 'bilinear':
                            observation = observed
                        if need:
                            row = {**record, 'width': rgb.shape[1], 'height': rgb.shape[0],
                                   'rgb_sha256': checksum, 'sampling': sampling, 'method': name,
                                   'metrics': quality(rgb, prediction, args.device, not args.basic_metrics)}
                            with previous.open('a') as output:
                                output.write(json.dumps(row, allow_nan=False) + '\n')
                            rows.append(row)
            for entry, model in loaded:
                name = entry['name']
                if (record['id'], sampling, name) in done:
                    continue
                neutral = 128/255 if entry['kind'] == 'legacy' else .5
                inputs = model_input(observation, args.matrix, neutral)
                torch.cuda.synchronize() if args.device.startswith('cuda') else None
                start = time.monotonic()
                value = infer(model, entry['kind'], inputs, sampling, args.device, args.tile)
                torch.cuda.synchronize() if args.device.startswith('cuda') else None
                seconds = time.monotonic() - start
                prediction = model_output(value, args.matrix, neutral)
                row = {**record, 'width': rgb.shape[1], 'height': rgb.shape[0], 'rgb_sha256': checksum,
                       'sampling': sampling, 'method': name, 'inference_seconds': seconds,
                       'metrics': quality(rgb, prediction, args.device, not args.basic_metrics)}
                with previous.open('a') as output:
                    output.write(json.dumps(row, allow_nan=False) + '\n')
                rows.append(row)
            write_json(args.output / 'report.json', {'status': 'running', 'rows': len(rows),
                'results': summarize(rows), 'published': PUBLISHED,
                'exact_published_protocol_verified': False})
            print(json.dumps({'image': record['id'], 'sampling': sampling, 'rows': len(rows),
                              'elapsed_seconds': time.monotonic()-started}), flush=True)
    write_json(args.output / 'report.json', {'status': 'complete', 'rows': len(rows),
        'results': summarize(rows), 'published': PUBLISHED, 'exact_published_protocol_verified': False,
        'elapsed_this_attempt_seconds': time.monotonic()-started})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--mode', choices=['calibrate', 'evaluate'], default='evaluate')
    parser.add_argument('--datasets', default='kodak,uhd240')
    parser.add_argument('--models', default='')
    parser.add_argument('--sampling', default='cosited_point')
    parser.add_argument('--matrix', choices=list(MATRICES), default='paper_rounded')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tile', type=int, default=512)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--basic-metrics', action='store_true')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.mode == 'calibrate':
        calibration(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
