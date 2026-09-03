# Research benchmark

The expanded benchmark is configured in
`research/configs/benchmark_full.json`; ablations are preregistered in
`research/configs/ablations.json`. See `research/PROTOCOL.md` for the evidence
and reporting rules.

Typical local workflow:

```bash
python scripts/build_research_manifest.py \
  --dataset-root data/research_lossless \
  --train data/research_lossless/train \
  --validation data/research_lossless/validation \
  --test data/research_lossless/test \
  --output research/manifests/lossless_full.jsonl

python scripts/run_ablation_matrix.py \
  --config research/configs/ablations.json

python scripts/research_benchmark.py \
  --config research/configs/benchmark_full.json
```

No publication-scale result is prefilled. The repository currently contains no
lossless training/test corpus, so generating a large numerical result without a
new local dataset would fabricate evidence. Tests generate small lossless images
only to verify the protocol machinery.

V7 uses the same manifest, degradation definitions, and reconstruction metrics.
Its configs are `v7.json`, `v7_ablations.json`, `v7_evaluation.json`, and the
disabled-by-default `v7_self_train.json`. The dedicated evaluator adds
uncertainty maps, interval coverage, correlations, and risk-coverage curves;
the common benchmark also accepts learned-method entries with `"type": "v7"`
and a `"mode"` of `"mean"` or `"safe"`.

The committed procedural audit can be regenerated without network access:

```bash
python scripts/generate_research_fixture.py
python scripts/build_research_manifest.py \
  --dataset-root research/fixtures/lossless \
  --train research/fixtures/lossless/train \
  --validation research/fixtures/lossless/validation \
  --test research/fixtures/lossless/test \
  --output research/manifests/fixture.jsonl
python scripts/run_ablation_matrix.py \
  --config research/configs/ablations_smoke.json --device cpu
CHROMA_FFMPEG=/path/to/ffmpeg python scripts/research_benchmark.py \
  --config research/configs/benchmark_smoke.json
```

Its current output contains 1,024 method-image-condition records over four
held-out fixture images: all 12 synthetic siting/filter combinations, four JPEG
qualities, four settings each for H.264, HEVC, and AV1, six classical methods,
V6, five learned smoke models, and 32 boundary sheets. All 28 configured
condition groups completed. If `CHROMA_FFMPEG` is unset and FFmpeg is not on
`PATH`, the video conditions are preserved as structured skips.
