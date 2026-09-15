#!/usr/bin/env python3
"""Compare accelerated metrics with independent CPU implementations and time 8K."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_li2026 import quality, quality_cpu, read_image, records, write_json
from chroma.li2026 import reconstruct
from chroma.li2026_metrics import lab, delta_e
from chroma.research_metrics import srgb_to_lab, delta_e_ciede2000_lab

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
results = []
rng = np.random.default_rng(44)
for size in [(128, 192), (600, 801)]:
    reference = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    candidate = np.clip(reference.astype(float)+rng.normal(0, 5, reference.shape), 0, 255)
    start = time.monotonic()
    expected = quality_cpu(reference, candidate, 'cuda')
    cpu_seconds = time.monotonic()-start
    start = time.monotonic()
    actual = quality(reference, candidate, 'cuda')
    gpu_seconds = time.monotonic()-start
    errors = {key: abs(actual[key]-expected[key]) for key in actual}
    for key, error in errors.items():
        if error > (1e-6 if key == 'fsim_luma' else 1e-9):
            raise ValueError(f'{key} changes with acceleration: {error}')
    results.append({'shape': size, 'errors': errors, 'cpu_seconds': cpu_seconds, 'gpu_seconds': gpu_seconds})
lab_a = rng.normal(0, 40, (1000, 3))
lab_b = rng.normal(0, 40, (1000, 3))
lab_a[:10, 1:] = 0
lab_b[10:20, 1:] = 0
expected = delta_e_ciede2000_lab(lab_a, lab_b)
actual = delta_e(torch.tensor(lab_a, device='cuda'), torch.tensor(lab_b, device='cuda')).cpu().numpy()
np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
root = Path('/ocean/projects/cis260224p/shared/brios/data/li2026')
rgb = read_image(records(root, 'uhd240_8k')[0])
start = time.monotonic()
candidate, _, _ = reconstruct(rgb)
transform_seconds = time.monotonic()-start
start = time.monotonic()
metrics = quality(rgb, candidate, 'cuda')
metric_seconds = time.monotonic()-start
report = {'status': 'passed', 'fixtures': results, 'full_8k': {
    'shape': list(rgb.shape), 'transform_seconds': transform_seconds,
    'metric_seconds': metric_seconds, 'metrics': metrics}}
write_json(Path(sys.argv[1]), report)
print(json.dumps(report), flush=True)
