# Dataset manifests

Publication manifests belong in this directory. Each JSONL line contains an
image identifier, split, relative path, SHA-256 digest, dimensions, original
mode, image format, and source group. The companion `.meta.json` records split
counts, skipped files, and the manifest's own digest.

`lossless_full.jsonl` is intentionally not fabricated. Create it only after a
local lossless corpus has been arranged as completely disjoint `train`,
`validation`, and `test` directories:

```bash
python scripts/build_research_manifest.py \
  --dataset-root data/research_lossless \
  --train data/research_lossless/train \
  --validation data/research_lossless/validation \
  --test data/research_lossless/test \
  --output research/manifests/lossless_full.jsonl
```

The builder verifies both decoded format and mode, rejects JPEG/WebP,
grayscale, and paletted inputs by default, and aborts if identical bytes occur in
different splits. Commit the resulting JSONL and metadata alongside the
benchmark's `per_image.jsonl`, `report.json`, and qualitative index.
