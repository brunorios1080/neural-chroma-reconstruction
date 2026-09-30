# Matched V5 and Prism MSE run

The V5 run in this experiment starts from random weights and uses the original
V5 U-Net generator architecture. It trains with **chroma MSE only**. The
generator's luminance output is replaced with the observed luminance, matching
the deterministic Prism models' output rule. This run has no discriminator.
It is a matched-objective architecture comparison, not a continuation of the
historical V5 GAN run.

The trainer reads the completed Prism U-Net MSE run's `manifest.jsonl` and
`config.snapshot.json`. It therefore uses the same COCO train/validation split,
12 degradations, epoch-addressed crops, augmentation, 256-pixel crop, batch size
32, seed 2026, AdamW settings, cosine schedule, and 100-epoch horizon.

Per-epoch training and validation chroma MSE are written to `history.jsonl`.
Every epoch has an `epoch_NNN.pth` checkpoint. `best.pth` is selected by the
smallest validation chroma MSE, and `last.pth` is the latest checkpoint.
`config.snapshot.json` records the source protocol and its manifest hash.

The completed run has all 100 epoch checkpoints. Epoch 100 is best, with
validation chroma MSE `0.00006505821271125754`. The checkpoints, history,
summary, configuration, and manifest are published in
[`models/experiments/v5_prism_mse100`](../../../models/experiments/v5_prism_mse100).
The `.pth` files and manifest use Git LFS; run `git lfs pull` after checking
out the checkpoint branch to download their contents.

The original cluster output directory is
`/ocean/projects/cis260224p/shared/brios/checkpoints/v5_prism_mse100`.
Load it for inference through `chroma.v5_prism_mse.load_matched_v5` so the
luminance passthrough is applied. A generic V5 checkpoint loader returns only
the raw generator and would give different full-image metrics.

Submit or resume with:

```bash
bash scripts/bridges2/submit_v5_prism_mse.sh
```
