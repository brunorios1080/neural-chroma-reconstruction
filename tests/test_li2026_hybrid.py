"""Hybrid invariants and epoch-boundary resume; run on a compute node."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn

from chroma.li2026 import reconstruct
from chroma.li2026_hybrid import (FORMAT, HybridDataset, HybridProtocol, HybridRefiner,
                                  initialize_hybrid, load_hybrid, prepare_baseline)
from chroma.li2026_hybrid_training import audit_sources, run_case, select_records
from chroma.prism_models import PrismArchitecture, PrismRefiner
from chroma.research_data import ManifestRecord, sha256_file


class OffsetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor([.05, -.02]).reshape(1, 2, 1, 1))

    def forward(self, value):
        return torch.cat((value[:, :1], value[:, 1:] + self.offset), 1)


class HybridTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(10)
        self.rgb = np.random.default_rng(11).integers(0, 256, (16, 16, 3), dtype=np.uint8)

    def batch(self, protocol=HybridProtocol()):
        baseline, keep = prepare_baseline(self.rgb, protocol)
        return torch.from_numpy(baseline).permute(2, 0, 1)[None], torch.from_numpy(keep)[None, None]

    def test_zero_gain_matches_all_analytic_variants_exactly(self):
        for transform in ['conventional', 'scaled_matrix', 'scaled_decode_first']:
            for sampling in ['cosited_point', 'center_box']:
                if transform == 'scaled_decode_first' and sampling == 'center_box':
                    continue
                for interpolation in ['bilinear', 'bicubic']:
                    protocol = HybridProtocol(transform, sampling, interpolation)
                    baseline, keep = self.batch(protocol)
                    expected, _, _ = reconstruct(self.rgb, sampling, interpolation, 'paper_rounded', transform)
                    model = HybridRefiner(OffsetBackbone(), {'kind': 'v6'}, protocol)
                    self.assertTrue(torch.equal(model(baseline, keep), baseline))
                    np.testing.assert_array_equal((baseline[0].permute(1, 2, 0).numpy()*255).round(), expected)
                    if transform == 'scaled_decode_first':
                        np.testing.assert_array_equal(expected[::2, ::2], self.rgb[::2, ::2])
                    self.assertEqual(int(keep.sum()), 64 if sampling == 'cosited_point' else 0)

    def test_nonzero_gain_protects_samples_and_luma(self):
        baseline, keep = self.batch()
        model = HybridRefiner(OffsetBackbone(), {'kind': 'v6'}, HybridProtocol())
        with torch.no_grad():
            model.correction_gain.fill_(.4)
        output = model(baseline, keep)
        self.assertTrue(torch.equal(output.masked_select(keep), baseline.masked_select(keep)))
        difference = output - baseline
        self.assertGreater(float(difference.abs().max()), 0)
        luminance = (difference * torch.tensor([.299, .587, .114])[None, :, None, None]).sum(1)
        self.assertLess(float(luminance.abs().max()), 1e-7)

    def test_gate_then_backbone_receive_finite_gradients(self):
        baseline, keep = self.batch()
        model = HybridRefiner(OffsetBackbone(), {'kind': 'v6'}, HybridProtocol())
        model(baseline, keep).square().mean().backward()
        self.assertGreater(float(model.correction_gain.grad.abs()), 0)
        self.assertEqual(float(model.backbone.offset.grad.abs().sum()), 0)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.correction_gain.fill_(.1)
        model(baseline, keep).square().mean().backward()
        self.assertGreater(float(model.backbone.offset.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(model.backbone.offset.grad).all())

    def test_prism_architectures_and_frozen_uncertainty(self):
        baseline, keep = self.batch()
        for representation in ['cartesian', 'polar']:
            for uncertainty in [False, True]:
                architecture = PrismArchitecture(width=4, depth=1, representation=representation,
                    uncertainty=uncertainty, detached_uncertainty=uncertainty and representation == 'cartesian',
                    conditioned=True)
                model = HybridRefiner(PrismRefiner(architecture),
                    {'kind': 'prism', 'architecture': asdict(architecture)}, HybridProtocol())
                self.assertTrue(torch.equal(model(baseline, keep), baseline))
                if uncertainty:
                    self.assertTrue(all(not p.requires_grad for p in model.backbone.uncertainty_head.parameters()))

    def test_invalid_protocol_and_mask_rejected(self):
        with self.assertRaises(ValueError):
            HybridProtocol(sampling='center_box').validate()
        baseline, keep = self.batch()
        model = HybridRefiner(OffsetBackbone(), {'kind': 'v6'}, HybridProtocol())
        with self.assertRaises(ValueError):
            model(baseline, keep.float())

    def test_flat_chroma_guard_preserves_gray_edges_and_constant_color(self):
        model = HybridRefiner(OffsetBackbone(), {'kind': 'v6'}, HybridProtocol(), flat_guard_radius=2)
        with torch.no_grad():
            model.correction_gain.fill_(.4)
        gray = torch.randint(20, 230, (1, 1, 16, 16)).float()/255
        keep = torch.zeros_like(gray, dtype=torch.bool)
        for offsets in [(0, 0, 0), (10/255, 0, -10/255)]:
            baseline = gray.expand(-1, 3, -1, -1) + torch.tensor(offsets)[None, :, None, None]
            self.assertTrue(torch.equal(model(baseline, keep), baseline))
        baseline, keep = self.batch()
        self.assertFalse(torch.equal(model(baseline, keep), baseline))

    def test_guard_metadata_survives_portable_loading(self):
        from chroma.li2026_hybrid import build_hybrid
        architecture = PrismArchitecture(width=4, depth=1)
        model = HybridRefiner(PrismRefiner(architecture),
            {'kind': 'prism', 'architecture': asdict(architecture)}, HybridProtocol(), flat_guard_radius=2)
        rebuilt = build_hybrid(model.metadata())
        self.assertEqual(rebuilt.flat_guard_radius, 2)
        baseline, keep = self.batch()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'hybrid.pth'
            torch.save({'format': FORMAT, 'metadata': model.metadata(), 'model': model.state_dict()}, path)
            loaded, _ = load_hybrid(path)
            self.assertEqual(loaded.flat_guard_radius, 2)
            self.assertTrue(torch.equal(model(baseline, keep), loaded(baseline, keep)))

    def fixture(self, root):
        records = []
        for index, split in enumerate(['train', 'train', 'validation', 'validation']):
            path = root / (str(index)+'.png')
            Image.fromarray(np.random.default_rng(index).integers(0, 256, (20, 20, 3), dtype=np.uint8)).save(path)
            records.append(ManifestRecord(str(index), path.name, split, sha256_file(path), 20, 20,
                                          'RGB', 'PNG', 'unit-test'))
        architecture = PrismArchitecture(width=4, depth=1)
        backbone = PrismRefiner(architecture)
        with torch.no_grad():
            backbone.tail.bias.fill_(.1)
        init = root / 'init.pth'
        torch.save({'format': 'prism-v1', 'architecture': asdict(architecture),
                    'model': backbone.state_dict(), 'epoch': 0}, init)
        entry = {'name': 'fixture', 'kind': 'prism', 'weights': str(init), 'sha256': sha256_file(init)}
        return records, entry

    def test_directory_zip_and_epoch_crops_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, _ = self.fixture(root)
            archive = root / 'fixture.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                for record in records:
                    stream.write(root / record.relative_path, record.relative_path)
            self.assertEqual(audit_sources(records, root), audit_sources(records, archive))
            direct = HybridDataset(records, root, 16, HybridProtocol(), 12, training=True)
            zipped = HybridDataset(records, archive, 16, HybridProtocol(), 12, training=True)
            for a, b in zip(direct[(1, 0)][:3], zipped[(1, 0)][:3]):
                self.assertTrue(torch.equal(a, b))
            self.assertFalse(torch.equal(direct[(1, 0)][1], direct[(2, 0)][1]))
            zipped.archive.close()

    def test_cross_split_duplicate_and_corrupted_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, _ = self.fixture(root)
            duplicate = replace(records[0], id='dup', relative_path='dup.png', split='validation')
            with self.assertRaisesRegex(ValueError, 'cross dataset splits'):
                audit_sources([*records, duplicate], root)
            (root / records[0].relative_path).write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                audit_sources(records, root)

    def test_selection_excludes_test_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, _ = self.fixture(root)
            records.append(replace(records[0], id='test', relative_path='test.png', sha256='test', split='test'))
            manifest = root / 'manifest.jsonl'
            manifest.write_text(''.join(json.dumps(asdict(r))+'\n' for r in records))
            training = {'seed': 5, 'crop_size': 16, 'train_images': 1, 'validation_images': 1}
            chosen = select_records(manifest, training)
            self.assertEqual(chosen, select_records(manifest, training))
            self.assertEqual([r.split for r in chosen], ['train', 'validation'])

    def test_resume_matches_uninterrupted_and_best_loads_without_init(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, entry = self.fixture(root)
            training = {'seed': 12, 'crop_size': 16, 'epochs': 2, 'batch_size': 2,
                        'workers': 0, 'backbone_lr': 1e-4, 'gain_lr': .003}
            case = {'name': 'fixture', 'model': 'fixture', 'protocol': asdict(HybridProtocol())}
            full = run_case(case, training, entry, records, root, root/'full', device='cpu')
            run_case(case, training, entry, records, root, root/'resumed', stop_after_epoch=1, device='cpu')
            resumed = run_case(case, training, entry, records, root, root/'resumed', resume=True, device='cpu')
            a = torch.load(root/'full/last.pth', weights_only=True)
            b = torch.load(root/'resumed/last.pth', weights_only=True)
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key], b['model'][key]), key)
            self.assertEqual(full['best_epoch'], resumed['best_epoch'])
            self.assertGreaterEqual(resumed['selected_validation']['mean_cpsnr_gain_db'], 0)
            with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                run_case(case, {**training, 'gain_lr': .01}, entry, records, root,
                         root/'resumed', resume=True, device='cpu')
            with self.assertRaises(FileExistsError):
                run_case(case, training, entry, records, root, root/'resumed', device='cpu')
            (root/'init.pth').unlink()
            portable, payload = load_hybrid(root/'resumed/best.pth')
            self.assertEqual(payload['format'], FORMAT)
            baseline, keep = self.batch()
            self.assertTrue(torch.isfinite(portable(baseline, keep)).all())

    @unittest.skipUnless(os.environ.get('HYBRID_REAL_CHECKPOINTS') == '1', 'cluster checkpoint preflight')
    def test_all_five_frozen_backbones_on_cuda(self):
        campaign = Path('/ocean/projects/cis260224p/shared/brios/evaluations/test2014_all_20260905/campaign.json')
        entries = {e['name']: e for e in json.loads(campaign.read_text())['models']}
        baseline, keep = (value.cuda() for value in self.batch())
        for name in ['v6', 'prism_residual', 'prism_polar', 'prism_polar_prob', 'prism_cartesian_prob']:
            with self.subTest(name=name):
                model = initialize_hybrid(entries[name], HybridProtocol()).cuda()
                prediction = model(baseline, keep)
                self.assertTrue(torch.equal(prediction, baseline))
                prediction.square().mean().backward()
                self.assertTrue(torch.isfinite(model.correction_gain.grad))
                with torch.no_grad():
                    model.correction_gain.fill_(.1)
                model.zero_grad(set_to_none=True)
                model(baseline, keep).square().mean().backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
