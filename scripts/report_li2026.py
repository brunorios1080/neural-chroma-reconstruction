#!/usr/bin/env python3
"""Audit completed Li-2026 shards and write a human-readable comparison.

Reads scalar results only; never runs inference or changes evaluation settings.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


NAMES = {
    'v5': 'V5', 'v5_1': 'V5.1', 'v6': 'V6', 'v7': 'V7',
    'prism_residual': 'Prism Residual', 'prism_polar': 'Prism Polar',
    'prism_polar_prob': 'Prism Polar Prob',
    'prism_cartesian_prob': 'Prism Cartesian Prob',
}
ORDER = list(NAMES)
TRANSFORMS = ('conventional', 'scaled_matrix', 'scaled_hybrid',
              'scaled_lifting', 'scaled_decode_first')
METRICS = {'cpsnr_rgb', 'mae_rgb_255', 'mse_rgb_255',
           'ciede2000', 'ssim_luma', 'fsim_luma'}


def mean(values):
    return statistics.mean(values)


def methods(sampling):
    transforms = TRANSFORMS if sampling == 'cosited_point' else TRANSFORMS[:2]
    return set(ORDER) | {a + '/' + b for a in transforms for b in ('bilinear', 'bicubic')}


def audited_rows(root):
    plan = json.loads((root / 'sharding.json').read_text())
    paths = [root / 'groups' / 'kodak'] + [root / 'shards' / str(i)
                                         for i in range(plan['shard_count'])]
    rows, completed, pending, files = [], [], [], []
    image_hashes, checkpoint_hashes, common_settings = {}, None, None
    seen = set()
    for path in paths:
        report_path = path / 'report.json'
        if not report_path.exists() or json.loads(report_path.read_text())['status'] != 'complete':
            pending.append(str(path.relative_to(root)))
            continue
        config = json.loads((path / 'configuration.json').read_text())
        settings = {name: config[name] for name in ('matrix', 'tile', 'halo', 'rounding',
                    'model_neutral_chroma', 'metric_definitions', 'source_campaign_sha256')}
        if common_settings is not None and settings != common_settings:
            raise ValueError('Evaluation settings differ between groups')
        common_settings = settings
        model_hashes = {m['name']: m['sha256'] for m in config['models']}
        if set(model_hashes) != set(ORDER):
            raise ValueError('Unexpected checkpoint set: ' + str(path))
        if checkpoint_hashes is not None and checkpoint_hashes != model_hashes:
            raise ValueError('Checkpoints differ between groups')
        checkpoint_hashes = model_hashes
        expected = {(r['dataset'], r['id'], s, m) for r in config['manifest']
                    for s in config['sampling'] for m in methods(s)}
        content = (path / 'rows.jsonl').read_bytes()
        entries = [json.loads(line) for line in content.splitlines()]
        actual = {(r['dataset'], r['id'], r['sampling'], r['method']) for r in entries}
        if actual != expected or len(actual) != len(entries) or actual & seen:
            raise ValueError('Missing, duplicate, or unexpected image/method pair: ' + str(path))
        if path.parent.name == 'shards':
            expected_ids = {key for key, shard in plan['membership'].items() if shard == int(path.name)}
            if {r['id'] for r in entries} != expected_ids:
                raise ValueError('Shard membership differs from frozen partition')
        for row in entries:
            if set(row['metrics']) != METRICS or not all(math.isfinite(v) for v in row['metrics'].values()):
                raise ValueError('Invalid metrics')
            key = (row['dataset'], row['id'])
            fingerprint = (row['rgb_sha256'], row['width'], row['height'])
            if key in image_hashes and image_hashes[key] != fingerprint:
                raise ValueError('Ground truth differs between methods')
            image_hashes[key] = fingerprint
        rows.extend(entries)
        seen.update(actual)
        completed.append(str(path.relative_to(root)))
        files.append({'path': str((path / 'rows.jsonl').relative_to(root)),
                      'sha256': hashlib.sha256(content).hexdigest(), 'rows': len(entries)})
    if not pending and len(rows) != 5040:
        raise ValueError('Expected 5040 evaluated image/method pairs')
    audit = {'status': 'complete' if not pending else 'partial', 'rows': len(rows),
             'unique_images': len(image_hashes), 'completed': completed, 'pending': pending,
             'result_files': files, 'checkpoint_hashes': checkpoint_hashes,
             'checks': ['exact image/method coverage for each completed group',
                        'no duplicate image/method pairs', 'same ground-truth RGB hash across methods',
                        'same eight checkpoint hashes', 'consistent transform, model, and metric settings',
                        'six finite metrics per result'],
             'exact_published_protocol_verified': False}
    return rows, audit


def generate(root):
    rows, audit = audited_rows(root)
    if audit['status'] == 'complete':
        fields = ['dataset', 'id', 'resolution', 'sampling', 'method', 'width', 'height', 'rgb_sha256']
        fields += sorted(METRICS)
        with (root / 'per_image.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for row in sorted(rows, key=lambda r: (r['dataset'], r['id'], r['sampling'], r['method'])):
                writer.writerow({**{name: row.get(name, '') for name in fields[:8]}, **row['metrics']})
    groups = defaultdict(dict)
    for row in rows:
        groups[(row['dataset'], row['sampling'], row['method'])][row['id']] = row
    scores = {key: mean(r['metrics']['cpsnr_rgb'] for r in values.values())
              for key, values in groups.items()}
    lines = ['# Comparison with Li et al. (2026)', '',
        '**Status: ' + audit['status'] + '.** ' + str(audit['rows']) +
        ' completed image/method pairs from ' + str(audit['unique_images']) + ' images.', '',
        ('All requested job groups are finalized.' if audit['status'] == 'complete' else
         'This report includes finalized job groups only; running groups may have additional saved results.'), '',
        'This is an independent implementation of the described experiment. '
        'The exhaustive color-transform test exactly matches Table 1. '
        'Image-table reproduction remains unverified because sampling, borders, '
        'lifting/interpolation interfaces, and metric settings are incompletely specified. '
        'Published values and our measurements must be interpreted separately.', '',
        'Source: [Li, Zhang, and Huang, IET Image Processing (2026)]'
        '(https://doi.org/10.1049/ipr2.70338). '
        'UHD images: [the authors\' Zenodo record](https://doi.org/10.5281/zenodo.17649711).', '']
    for dataset, sampling in [('uhd240', 'cosited_point'), ('kodak', 'cosited_point'), ('kodak', 'center_box')]:
        key = (dataset, sampling)
        if key + ('v6',) not in groups:
            continue
        ids = set(groups[key + ('v6',)])
        expected = 240 if dataset == 'uhd240' else 24
        if len(ids) != expected:
            lines.extend(['## ' + dataset + ' / ' + sampling, '',
                          'Partial coverage (' + str(len(ids)) + '/' + str(expected) +
                          ' images). A final ranking is withheld.', ''])
            continue
        baseline = scores[key + ('conventional/bilinear',)]
        comparators = [m for m in methods(sampling) if m.startswith('scaled_')]
        best = max(comparators, key=lambda m: scores[key + (m,)])
        strongest = scores[key + (best,)]
        reference = groups[key + (best,)]
        lines.extend(['## ' + dataset + ' / ' + sampling, '',
            'All ' + str(expected) + ' native images. RGB CPSNR averages per-image dB; higher is better.', '',
            '| Method | RGB CPSNR (dB) | Gain vs bilinear | Gain vs best independent scaled method | Wins vs scaled |',
            '|---|---:|---:|---:|---:|',
            '| Conventional bilinear | %.5f | — | %.5f | — |' % (baseline, baseline - strongest),
            '| ' + best + ' | %.5f | %+.5f | — | — |' % (strongest, strongest - baseline)])
        for method in ORDER:
            value = scores[key + (method,)]
            entries = groups[key + (method,)]
            if set(entries) != ids or set(reference) != ids:
                raise ValueError('Unpaired comparator')
            wins = sum(entries[i]['metrics']['cpsnr_rgb'] > reference[i]['metrics']['cpsnr_rgb'] for i in ids)
            lines.append('| %s | %.5f | %+.5f | %+.5f | %d/%d |' %
                         (NAMES[method], value, value - baseline, value - strongest, wins, expected))
        lines.extend(['', 'The strongest independent comparator is selected by dataset mean, '
                      'not separately for each image. This is not an author-code comparison.', '',
                      '### Learned models: secondary metrics', '',
                      '| Model | CIEDE2000 (lower) | RGB MAE (lower) | Luma SSIM (higher) | Luma FSIM (higher) |',
                      '|---|---:|---:|---:|---:|'])
        for method in ORDER:
            entries = groups[key + (method,)].values()
            metrics = [mean(r['metrics'][name] for r in entries)
                       for name in ('ciede2000', 'mae_rgb_255', 'ssim_luma', 'fsim_luma')]
            lines.append('| %s | %.5f | %.5f | %.7f | %.7f |' % (NAMES[method], *metrics))
        lines.extend(['',
                      '### All independent interpolation variants', '',
                      '| Method | RGB CPSNR | CIEDE2000 (lower) | RGB MAE (lower) | Luma SSIM | Luma FSIM |',
                      '|---|---:|---:|---:|---:|---:|'])
        for method in sorted(m for m in methods(sampling) if '/' in m):
            entries = groups[key + (method,)].values()
            lines.append('| %s | %.5f | %.5f | %.5f | %.7f | %.7f |' % (method, scores[key + (method,)],
                mean(r['metrics']['ciede2000'] for r in entries),
                mean(r['metrics']['mae_rgb_255'] for r in entries),
                mean(r['metrics']['ssim_luma'] for r in entries),
                mean(r['metrics']['fsim_luma'] for r in entries)))
        lines.append('')
        if sampling == 'cosited_point':
            published = {'uhd240': (50.49, 50.69, 54.40, 54.82),
                         'kodak': (47.33, 46.28, 48.31, 47.42)}[dataset]
            lines.extend(['Published CPSNR: conventional bilinear %.2f; conventional bicubic %.2f; '
                          'scaled/lifting bilinear %.2f; scaled/lifting bicubic %.2f dB.' % published,
                          'Our conventional bilinear differs from the published value by %+.5f dB.' %
                          (baseline - published[0]), ''])
    # Resolution-specific metrics are valuable, but only once all 80 images are present.
    resolution_rows = []
    for resolution in ('4K', '6K', '8K'):
        for method in sorted(methods('cosited_point')):
            entries = [r for r in rows if r['dataset'] == 'uhd240' and
                       r.get('resolution') == resolution and r['method'] == method]
            if len(entries) == 80:
                resolution_rows.append({'resolution': resolution, 'method': method, 'images': 80,
                                        **{m: mean(r['metrics'][m] for r in entries) for m in sorted(METRICS)}})
    if resolution_rows:
        with (root / 'by_resolution.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(resolution_rows[0]))
            writer.writeheader()
            writer.writerows(resolution_rows)
    lines.extend(['## Interpretation limits', '',
        '- Models receive the same conventional bilinear observation. Scaled methods alter the encoder '
        'representation as described in the paper; equal bitrate and equal intermediate bit depth are not established.',
        '- Kodak centered-box sampling is a sensitivity experiment, not the paper\'s confirmed protocol.',
        '- No model was trained or selected using these test scores. Checkpoints were frozen before evaluation.',
        '- The legacy training-image manifests remain unknown. Prior overlap-audit fields in checkpoint '
        'metadata refer to the earlier COCO campaign; they do not certify absence of UHD/Kodak training overlap.',
        '- These RGB scores differ in metric and degradation from the earlier COCO chroma-PSNR experiment.', ''])
    preflight = json.loads((root / 'preflight.json').read_text())
    model_sizes = {m['name']: m['parameters'] for m in preflight['models']}
    lines.extend(['## Model size', '',
                  'The paper\'s proposed method uses analytically computed transform coefficients and '
                  'interpolation, with zero trainable parameters and no neural-network checkpoint.', '',
                  '| Model | Trainable parameters | FP32 weights only (MiB) |',
                  '|---|---:|---:|'])
    for name in ORDER:
        count = model_sizes[name]
        lines.append('| %s | %s | %.3f |' % (NAMES[name], format(count, ','), count * 4 / 1024**2))
    lines.extend(['', 'Weight storage excludes activations, optimizer state, and other checkpoint contents. '
                  'These are not runtime-memory or measured-speed comparisons.', ''])
    if audit['status'] == 'complete':
        lines.extend(['## Downloads', '',
                      '- [Aggregate metrics](comparison.csv)',
                      '- [Per-image metrics](per_image.csv)',
                      '- [Results by resolution](by_resolution.csv)',
                      '- [Chart (PDF)](comparison_plot.pdf)',
                      '- [Coverage and provenance audit](audit.json)', ''])
    (root / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (root / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({k: audit[k] for k in ('status', 'rows', 'unique_images', 'pending')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    generate(args.root)
