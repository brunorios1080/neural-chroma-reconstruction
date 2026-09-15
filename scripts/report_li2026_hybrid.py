#!/usr/bin/env python3
"""Audit and summarize a frozen hybrid pilot using its saved per-image rows."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root):
    hashes = json.loads((root/'code_sha256.json').read_text())
    for relative, expected in hashes.items():
        if sha256(root/'code'/relative) != expected:
            raise ValueError('Frozen code changed: '+relative)
    configs = list((root/'code/research/configs').glob('li2026_hybrid*pilot.json'))
    if len(configs) != 1:
        raise ValueError('Expected exactly one frozen pilot configuration')
    config = json.loads(configs[0].read_text())
    cases = {case['name']: case for case in config['cases']}
    results = []
    selection = None
    for path in sorted((root/'runs').glob('*/report.json')):
        run = path.parent
        report = json.loads(path.read_text())
        rows = json.loads((run/'validation_best.json').read_text())
        records = [json.loads(line) for line in (run/'manifest.jsonl').read_text().splitlines() if line.strip()]
        current = [(r['id'], r['split'], r['sha256']) for r in records]
        if selection is not None and current != selection:
            raise ValueError('Cases used different data selections')
        selection = current
        train_ids = {r['id'] for r in records if r['split'] == 'train'}
        val_ids = {r['id'] for r in records if r['split'] == 'validation'}
        if train_ids & val_ids or len(rows) != len(val_ids) or {r['id'] for r in rows} != val_ids:
            raise ValueError('Validation row coverage or split failure')
        selected = report['selected_validation']
        for family in ['baseline', 'hybrid']:
            for metric in ['mse_rgb_255', 'cpsnr_rgb', 'mae_rgb_255']:
                key = family+'_'+metric
                if not math.isclose(statistics.mean(r[key] for r in rows), selected[key], abs_tol=1e-10):
                    raise ValueError('Summary does not match per-image results: '+key)
        expected_epochs = config['training']['epochs']
        if report['completed_epochs'] != expected_epochs:
            raise ValueError('Training incomplete: '+run.name)
        wins = sum(r['hybrid_cpsnr_rgb'] > r['baseline_cpsnr_rgb'] for r in rows)
        if wins != selected['wins'] or selected['mean_cpsnr_gain_db'] < 0:
            raise ValueError('Selection or win count failure')
        if selected['epoch'] != report['best_epoch']:
            raise ValueError('Selected epoch mismatch')
        item = {'case': run.name, 'train_images': len(train_ids), 'validation_images': len(rows),
                'completed_epochs': report['completed_epochs'], 'selected_epoch': report['best_epoch'],
                'baseline_cpsnr': selected['baseline_cpsnr_rgb'],
                'hybrid_cpsnr': selected['hybrid_cpsnr_rgb'],
                'gain_db': selected['mean_cpsnr_gain_db'], 'wins': wins,
                'mse_reduction_percent': 100*(1-selected['hybrid_mse_rgb_255']/selected['baseline_mse_rgb_255']),
                'baseline_perfect': sum(r['baseline_mse_rgb_255'] == 0 for r in rows),
                'hybrid_perfect': sum(r['hybrid_mse_rgb_255'] == 0 for r in rows),
                'last_epoch_gain_db': report['last_validation']['mean_cpsnr_gain_db'],
                'best_checkpoint_sha256': sha256(run/'best.pth'),
                'flat_guard_radius': cases[run.name].get('flat_guard_radius', 0)}
        results.append(item)
    if len(results) != 4:
        raise ValueError('Expected all four pilot cases')
    payload = {'audit_passed': True, 'frozen_code_files': len(hashes), 'cases': results,
               'limitations': ['Validation-selected crop pilot, one seed; not an independent test.',
                   'CPSNR uses 1e-12 MSE floor for perfect crops; MAE/MSE are unmodified.',
                   'Independent analytic implementation; author image tables not exactly reproduced.']}
    (root/'audit.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    lines = ['# Hybrid COCO pilot', '',
             '256 training / 64 validation images, 128×128 crops, four epochs, one seed.',
             'Each row compares the selected hybrid to its own analytic baseline.', '',
             '| Case | Selected epoch | Baseline CPSNR | Hybrid CPSNR | Gain | Wins | MSE reduction |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['case']} | {r['selected_epoch']} | {r['baseline_cpsnr']:.4f} | "
                     f"{r['hybrid_cpsnr']:.4f} | {r['gain_db']:+.4f} dB | {r['wins']}/64 | "
                     f"{r['mse_reduction_percent']:.2f}% |")
    lines.extend(['', 'Epoch zero is the unmodified analytic baseline and is eligible for selection.',
                  'A zero selected gain can therefore mean trained checkpoints regressed.', '',
                  *payload['limitations'], '',
                  'Frozen code, source selections, source hashes, checkpoint provenance and all',
                  'per-image results are retained alongside this report. `audit.json` verifies',
                  'coverage, split identities, matched data, checkpoint hashes and summary arithmetic.'])
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.root), indent=2))
