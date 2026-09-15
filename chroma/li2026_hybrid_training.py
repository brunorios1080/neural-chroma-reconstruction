"""Bounded, reproducible fine-tuning of the independent analytic/neural hybrid."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import fcntl
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import time
import zipfile

from PIL import Image
import torch
from torch.utils.data import DataLoader

from .li2026_hybrid import (FORMAT, HybridDataset, HybridProtocol,
                            initialize_hybrid, load_hybrid, validate_hybrid)
from .prism_data import EpochSampler
from .prism_training import atomic_json, save_checkpoint
from .research_data import load_manifest, sha256_file


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def select_records(manifest, training):
    """Keep the existing split; selection depends on IDs, never image scores."""
    records = load_manifest(manifest)
    seen_ids, seen_paths, hash_splits = set(), set(), {}
    for record in records:
        path = PurePosixPath(record.relative_path)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe manifest path')
        if record.id in seen_ids or record.relative_path in seen_paths:
            raise ValueError('Duplicate manifest ID/path')
        seen_ids.add(record.id)
        seen_paths.add(record.relative_path)
        if record.sha256 in hash_splits and hash_splits[record.sha256] != record.split:
            raise ValueError('Identical source bytes cross dataset splits')
        hash_splits[record.sha256] = record.split
    selected = []
    for split, key in [('train', 'train_images'), ('validation', 'validation_images')]:
        candidates = [r for r in records if r.split == split and
                      min(r.width, r.height) >= training['crop_size']]
        candidates.sort(key=lambda r: digest([training['seed'], r.id]))
        count = int(training[key])
        if count < 1 or len(candidates) < count:
            raise ValueError('Insufficient eligible images for ' + split)
        selected.extend(candidates[:count])
    return selected


def audit_sources(records, source):
    """Hash encoded source bytes and verify metadata before any optimization."""
    source = Path(source)
    seen_ids, seen_paths, hash_splits = set(), set(), {}
    for record in records:
        path = PurePosixPath(record.relative_path)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe manifest path')
        if record.id in seen_ids or record.relative_path in seen_paths:
            raise ValueError('Duplicate source ID/path')
        if record.split not in {'train', 'validation'}:
            raise ValueError('Hybrid fine-tuning only accepts train/validation images')
        if record.sha256 in hash_splits and hash_splits[record.sha256] != record.split:
            raise ValueError('Identical source bytes cross dataset splits')
        seen_ids.add(record.id)
        seen_paths.add(record.relative_path)
        hash_splits[record.sha256] = record.split
    context = nullcontext(None) if source.is_dir() else zipfile.ZipFile(source)
    with context as archive:
        for record in records:
            if archive is None:
                path = (source / record.relative_path).resolve()
                if not path.is_relative_to(source.resolve()):
                    raise ValueError('Source escapes dataset directory')
                data = path.read_bytes()
            else:
                data = archive.read(record.relative_path)
            if hashlib.sha256(data).hexdigest() != record.sha256:
                raise ValueError('Source checksum mismatch: ' + record.id)
            with Image.open(io.BytesIO(data)) as image:
                if image.size != (record.width, record.height) or image.mode != record.mode:
                    raise ValueError('Source metadata mismatch: ' + record.id)
                image.verify()
    return {'images': len(records), 'encoded_sha256_verified': True,
            'train_validation_disjoint_ids_paths_hashes': True,
            'selection_sha256': digest([asdict(r) for r in records])}


def code_hashes():
    directory = Path(__file__).parent
    return {path.name: sha256_file(path) for path in sorted(directory.glob('*.py'))}


def run_case(case, training, entry, records, source, output, *, resume=False,
             stop_after_epoch=None, device='cuda'):
    """Save epoch zero as a valid fallback; select only on COCO validation CPSNR.

    Resume is at completed epoch boundaries. Epoch-addressed crops and a private
    loader generator make worker startup independent of model RNG state.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_case(case, training, entry, records, source, output, resume,
                         stop_after_epoch, device)


def _run_case(case, training, entry, records, source, output, resume, stop_after_epoch, device):
    protocol = HybridProtocol(**case['protocol'])
    protocol.validate()
    if training['epochs'] < 1 or training['batch_size'] < 1 or training['workers'] < 0:
        raise ValueError('Invalid training limits')
    if min(training['backbone_lr'], training['gain_lr']) <= 0:
        raise ValueError('Learning rates must be positive')
    # Audit inside this callable too: callers cannot bypass source verification.
    audit = audit_sources(records, source)
    provenance = {'case': case, 'training': training, 'initialization': entry,
                  'selection': [asdict(r) for r in records], 'code': code_hashes()}
    fingerprint = digest(provenance)
    checkpoint_path = output / 'last.pth'
    if not resume and any(p.name != '.lock' for p in output.iterdir()):
        raise FileExistsError('Run exists; choose a new output or --resume')
    if resume and not checkpoint_path.is_file():
        raise FileNotFoundError('Resume requires last.pth')
    torch.manual_seed(training['seed'])
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for the cluster pilot')
        torch.cuda.manual_seed_all(training['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    payload = None
    if resume:
        model, payload = load_hybrid(checkpoint_path, device)
        if payload['fingerprint'] != fingerprint:
            raise ValueError('Resume fingerprint mismatch (protocol/data/model/config/code)')
    else:
        model = initialize_hybrid(entry, protocol, case.get('flat_guard_radius', 0)).to(device)
    optimizer = torch.optim.Adam([
        {'params': [p for p in model.backbone.parameters() if p.requires_grad],
         'lr': training['backbone_lr']},
        {'params': [model.correction_gain], 'lr': training['gain_lr']},
    ])
    datasets = {split: HybridDataset([r for r in records if r.split == split], source,
                                    training['crop_size'], protocol, training['seed'],
                                    training=split == 'train')
                for split in ['train', 'validation']}
    sampler = EpochSampler(datasets['train'], training['seed'])
    loader_args = {'batch_size': training['batch_size'], 'num_workers': training['workers'],
                   'pin_memory': device.startswith('cuda'),
                   'persistent_workers': training['workers'] > 0}
    train_loader = DataLoader(datasets['train'], sampler=sampler,
                              generator=torch.Generator().manual_seed(training['seed']), **loader_args)
    val_loader = DataLoader(datasets['validation'], shuffle=False,
                            generator=torch.Generator().manual_seed(training['seed']+1), **loader_args)
    history, best_epoch, best_score, start = [], 0, -float('inf'), 0
    if payload:
        optimizer.load_state_dict(payload['optimizer'])
        torch.set_rng_state(payload['torch_rng'].cpu())
        if device.startswith('cuda'):
            torch.cuda.set_rng_state_all([s.cpu() for s in payload['cuda_rng']])
        history, best_epoch, best_score = payload['history'], payload['best_epoch'], payload['best_score']
        start = payload['epoch'] + 1
    else:
        atomic_json(output / 'provenance.json', {**provenance, 'fingerprint': fingerprint,
                    'audit': audit, 'source': str(source), 'device': device,
                    'torch_version': str(torch.__version__),
                    'original_v6_training_images_unknown': entry['kind'] == 'legacy'})
        (output / 'manifest.jsonl').write_text(''.join(json.dumps(asdict(r))+'\n' for r in records))
    last_epoch = min(training['epochs'], stop_after_epoch) if stop_after_epoch is not None else training['epochs']
    for epoch in range(start, last_epoch + 1):
        started = time.monotonic()
        training_loss, images = 0., 0
        if epoch:
            model.train()
            sampler.set_epoch(epoch)
            for baseline, target, keep, _ in train_loader:
                baseline, target, keep = baseline.to(device), target.to(device), keep.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(baseline, keep)
                loss = (prediction - target).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite RGB MSE')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                training_loss += float(loss.detach()) * len(target)
                images += len(target)
        summary, rows = validate_hybrid(model, val_loader, device)
        improved = summary['hybrid_cpsnr_rgb'] > best_score
        if improved:
            best_score, best_epoch = summary['hybrid_cpsnr_rgb'], epoch
        history.append({'epoch': epoch, 'train_rgb_mse': training_loss/images if images else None,
                        'seconds': time.monotonic()-started, **summary})
        state = {'format': FORMAT, 'metadata': model.metadata(), 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'fingerprint': fingerprint,
                 'initialization': entry, 'history': history, 'best_epoch': best_epoch,
                 'best_score': best_score, 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
                 'validation_rows': rows}
        # last.pth is the authoritative resume transaction. Rebuild derived
        # reports/best from it when resuming an interrupted checkpoint write.
        previous_best = payload['best_state'] if payload else None
        state['best_state'] = ({key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
                               if improved else previous_best)
        state['best_rows'] = rows if improved else payload['best_rows']
        save_checkpoint(checkpoint_path, state)
        payload = state
        _write_derived(output, payload)
        print(json.dumps({'case': case['name'], **history[-1]}, allow_nan=False), flush=True)
    if payload is None:
        raise ValueError('No checkpoint produced')
    _write_derived(output, payload)
    return json.loads((output / 'report.json').read_text())


def _write_derived(output, payload):
    best = payload['history'][payload['best_epoch']]
    portable = {'format': FORMAT, 'metadata': payload['metadata'], 'model': payload['best_state'],
                'epoch': payload['best_epoch'], 'fingerprint': payload['fingerprint'],
                'initialization': payload['initialization'], 'validation': best}
    save_checkpoint(output / 'best.pth', portable)
    atomic_json(output / 'history.json', payload['history'])
    atomic_json(output / 'validation_last.json', payload['validation_rows'])
    atomic_json(output / 'validation_best.json', payload['best_rows'])
    atomic_json(output / 'report.json', {'completed_epochs': payload['epoch'],
                'best_epoch': payload['best_epoch'], 'selected_validation': best,
                'last_validation': payload['history'][-1],
                'validation_selection_includes_epoch_zero': True,
                'scope': 'COCO crop pilot; no UHD/Kodak evaluation or author-code comparison'})
