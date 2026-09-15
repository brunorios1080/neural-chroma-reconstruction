#!/usr/bin/env python3
"""Check complete-vs-tiled inference and metric identity on real checkpoints."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_li2026 import TRAINED, infer, quality, write_json
from chroma.model_comparison import load_candidate

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
campaign = json.loads(Path(sys.argv[1]).read_text())
inputs = np.random.default_rng(4).uniform(.2, .8, (192, 256, 3)).astype(np.float32)
results = []
for entry in campaign['models']:
    if entry['name'] not in TRAINED:
        continue
    model, metadata = load_candidate(entry, 'cuda')
    whole = infer(model, entry['kind'], inputs, 'cosited_point', 'cuda', tile=0)
    tiled = infer(model, entry['kind'], inputs, 'cosited_point', 'cuda', tile=64)
    difference = float(np.max(np.abs(whole - tiled)))
    if difference > 5e-6:
        raise ValueError(f"Tiling changes {entry['name']}: {difference}")
    results.append({'name': entry['name'], 'tile_max_abs_difference': difference, **metadata})
    del model
reference = np.random.default_rng(17).integers(0, 255, (192, 256, 3), dtype=np.uint8)
candidate = np.minimum(reference.astype(np.float64) + 1, 255)
metrics = quality(reference, candidate, 'cuda')
assert abs(metrics['cpsnr_rgb'] - 48.1308036087) < 1e-8
assert metrics['mae_rgb_255'] == 1
write_json(Path(sys.argv[2]), {'status': 'passed', 'models': results, 'metric_check': metrics,
                             'torch': torch.__version__, 'device': torch.cuda.get_device_name()})
print(json.dumps(results), flush=True)
