# Prism training and evaluation

Prism is a new experiment family. These models train from scratch in their own
directories and use `prism-v1` checkpoints. Original V5, V5.1, V6, and V7 weights
are not compatible and are never used as implicit starting points.

## Experiments

| Name | Architecture / change | Matched comparison |
| --- | --- | --- |
| `prism_residual` | V6-style Cartesian residual trunk; zero-initialized output | Shared reference |
| `prism_mse` | Same model, MSE reconstruction | L1 versus MSE |
| `prism_edge` | Same model, L1 + 0.2 × chroma-gradient L1 | Boundary supervision |
| `prism_naf` | NAF-style blocks, Cartesian residual output | Restoration backbone |
| `prism_polar` | Deterministic polar output with smooth positive amplitude | Cartesian versus polar |
| `prism_polar_prob` | Same polar model, separate uncertainty heads with ramped joint losses | Uncertainty's effect on reconstruction |
| `prism_cartesian_prob` | Cartesian residual mean; detached polar uncertainty heads | Uncertainty without auxiliary gradients changing reconstruction |
| `prism_conditioned` | Residual model plus known siting/filter conditioning | Known versus inferred degradation |
| `prism_forward` | Residual model + 0.1 × measurement-consistency L1 | Measurement consistency |
| `prism_polar_forward` | Probabilistic polar model + measurement consistency | Same forward-loss ablation on polar model |
| `prism_unet` | Chroma-residual U-Net, bilinear decoder upsampling, unchanged Y | U-Net reconstruction control |
| `prism_gan` | Same U-Net with a conditional PatchGAN and small adversarial weight | Adversarial effect |

Every model preserves luminance. Residual/NAF trunks default to width 64 and
depth 8; U-Nets start at width 32. Training settings and samples are matched,
but different architectures do not have identical parameter counts or runtime.

## Shared experimental rules

- Analytical floating-point Y/Cr/Cb with neutral chroma 0.5, 256-pixel crops,
  batch 16, seed 2026, AdamW at 1e-4, cosine decay to 1e-6 over 100 epochs.
- Float32 training with gradient checks/clipping. TF32 is configurable; mixed
  precision is deliberately not part of this first controlled suite.
- A sampler sends `(epoch, image_index)` to workers. Persistent workers receive
  the new epoch for every task; crop/degradation selection refreshes each epoch
  and is identical across experiments. Flips/rotations precede subsampling.
- The same 12 siting/filter conditions are used everywhere. Epoch validation
  assigns one fixed condition per validation image, evenly by index, for cost.
  Set `validation_all_degradations` to true for the full cross product. The
  standalone evaluator always runs all configured conditions on every image.
- Each checkpoint is selected by minimum **validation chroma L1**, including
  MSE, edge, and adversarial runs. Test data never selects a checkpoint.
- Validation uses the same final [0,1] clipping for all models and also reports
  the fraction of raw out-of-range predictions. Training losses use raw outputs.
- Logs include chroma/RGB PSNR and SSIM, full/chroma/luma L1, bilinear metrics,
  paired average gains, per-degradation metrics, learning rate, and gradient norm.
  SSIM handles identical black images correctly.
- Uncertainty losses start after five epochs and ramp over ten. For
  `prism_cartesian_prob`, both features and predicted means are detached from
  auxiliary losses. Reconstruction continues to train normally.
- The conditional discriminator receives observed Y/Cr/Cb plus candidate Cr/Cb.
  Its 0.001 adversarial weight ramps after five reconstruction-only epochs.
- Confidence is computed from predictions only. `safe` mode blends the learned
  correction toward bilinear. Uncertainty intervals are nominal distributions;
  empirical coverage must be evaluated before making calibration claims.

Polar amplitude uses a softplus transform initialized to the bilinear radius
(within 5e-6 at neutral gray). It has no hard zero-radius clipping during
optimization. Smooth functions can still saturate numerically for extreme
activations; Cartesian controls and finite-gradient checks remain necessary.

The conditioned model requires the actual siting/filter at inference. Its result
is a **known-degradation** experiment, not a blind codec-restoration claim.
Forward consistency currently models synthetic linear subsampling; compressed
observations require a noise/quantization-aware extension.

## Configurations

- `research/configs/prism.json`: lossless scientific corpus. Uses the repository's
  `lossless_full.jsonl` manifest, which must be built from separate splits.
- `research/configs/prism_coco.json`: inherits the same experiments for COCO JPEG
  development. It is not a lossless-source scientific evaluation.
- `research/configs/prism_coco_test2014.json`: held-out COCO 2014 test evaluation,
  with bilinear interpolation as the sole classical baseline.
- `research/configs/prism_smoke.json`: two tiny fixture epochs; exercises all loss
  branches immediately. These outputs are software checks, not model results.
- `research/configs/prism_gpu_smoke.json`: two COCO epochs at production model,
  crop, and batch sizes; activates uncertainty/GAN losses immediately. Its
  `max_images: 100` limits the Bridges-2 launcher to a small GPU pipeline check.
  These runs do not provide scientific results.

JSON `extends` paths are relative to the containing config. Dataset/output paths
inside configs are relative to the repository. CLI paths are relative to the
shell's current directory.

## Bridges-2

Use the existing `scripts/bridges2/setup.sh` environment setup first if needed.
The launch wrapper uses PSC's Python directly for listing so the old system
`python3` on a login node does not break submission. `NCR_PYTHON` can override it.
The wrapper explicitly uses `--export=ALL` so `NCR_*` settings reach jobs even
when the submitting shell has `SBATCH_EXPORT=NONE`.

From the repository, preview submission (no jobs submitted):

```bash
bash scripts/bridges2/submit_prism.sh --dry-run
```

Submit a small pipeline check on two models:

```bash
NCR_WALLTIME=00:30:00 \
  bash scripts/bridges2/submit_prism.sh --config research/configs/prism_gpu_smoke.json \
  prism_residual prism_polar_prob
```

Submit selected full experiments, or omit names to submit all twelve:

```bash
bash scripts/bridges2/submit_prism.sh \
  prism_residual prism_polar prism_polar_prob prism_cartesian_prob

# All twelve independent experiments:
bash scripts/bridges2/submit_prism.sh
```

These commands submit a Slurm array with one L40S per experiment and at most
two concurrent jobs. Default walltime is eight hours per task: all twelve tasks
have a combined upper bound of **96 GPU-hours** at that limit. Concurrency does
not reduce total GPU-hours. Start with the small checks and selected comparisons.
Changing `NCR_GPU` changes hardware/cost; the wrapper does not choose a new rate.

The job stages the existing COCO ZIP on node-local storage, hashes images,
deduplicates exact content, and assigns the same 94% train / 4% validation /
2% test split in every task. At least 50 distinct eligible images are required.
The manifest is copied into each run's output directory before training.
`NCR_MAX_IMAGES` overrides the config's `max_images` when supplied. Limited runs
check the manifest count before training, including externally supplied manifests,
and refuse counts above the limit or below 50.

Default outputs:

```text
/ocean/projects/cis260224p/shared/$USER/checkpoints/prism/prism_coco/<model>/
/ocean/projects/cis260224p/shared/$USER/checkpoints/prism/prism_gpu_smoke_smoke100/<model>/
```

Other overrides: `NCR_OUTPUT_ROOT`, `NCR_PROJECT_ROOT`, `NCR_DATASET_ARCHIVE`,
`NCR_COCO_METADATA`, `NCR_EPOCHS`, `NCR_STOP_AFTER_EPOCH`, `NCR_BATCH_SIZE`, `NCR_WORKERS`, `NCR_CPUS`,
`NCR_GPU`, `NCR_MEMORY`, `NCR_WALLTIME`, `NCR_ACCOUNT`, `NCR_PARTITION`, and
`NCR_MAX_CONCURRENT`. All comparison tasks should receive the same overrides.
`NCR_DEPENDENCY=afterok:<smoke_job_id>` gates a submission on a successful smoke
job. Dependent jobs are cancelled if that dependency becomes impossible.

For an existing lossless corpus, supply its manifest/root and use the lossless
configuration; this does not stage the COCO archive:

```bash
NCR_MANIFEST=/absolute/path/lossless_full.jsonl \
NCR_DATASET_ROOT=/absolute/path/lossless_dataset \
  bash scripts/bridges2/submit_prism.sh --config research/configs/prism.json
```

## Monitoring and resuming

The submission output maps array task indices to model names. Slurm logs use
`slurm_logs/prism-<array_job_id>_<task_index>.out` and `.err`. For example, if
your actual array ID is 12345, task zero's training log is:

```bash
tail -n 20 -f slurm_logs/prism-12345_0.out
```

Alternatively, watch any model's structured per-epoch log directly:

```bash
tail -n 5 -f /ocean/projects/cis260224p/shared/$USER/checkpoints/prism/prism_coco/prism_residual/history.jsonl
```

Read a compact report across the four core models (safe on a login node):

```bash
/opt/packages/AI/pytorch_26.05-py3/bin/python3 scripts/review_prism.py \
  --output-root /ocean/projects/cis260224p/shared/$USER/checkpoints/prism/prism_coco
```

The report includes the latest epoch, best validation chroma L1 and its epoch,
recent-five-epoch mean L1, latest RGB PSNR/SSIM, and chroma PSNR gain over bilinear.
It is marked ready only when all four reach epoch 25 with matching manifests.
It does not select or submit further experiments automatically. Review the full
curves and per-degradation metrics before expanding the suite; these COCO runs
remain development experiments rather than lossless-source research evidence.

Each run saves atomic `last.pth` and `best.pth`, `history.jsonl`, `summary.json`,
the resolved configuration, manifest, and code/package provenance. Checkpoints
include optimizer, cosine schedule, discriminator (where applicable), and RNG
state. A run lock prevents two workers writing the same model directory.

Submitting the same task again automatically resumes `last.pth`. Resume checks
the experiment, loss, training settings, and content-addressed manifest. Changing
batch size, learning rate, epochs/scheduler horizon, or dataset requires a new
output root. Runtime worker count and device may change; bitwise agreement across
hardware is not promised. `--stop-after-epoch` stops a local test early without
changing its planned schedule, and `--resume auto` continues it.
For the four-model initial comparison, set `NCR_STOP_AFTER_EPOCH=25` when
submitting. Leave `NCR_EPOCHS` unset to retain the 100-epoch cosine schedule.
Resubmit without `NCR_STOP_AFTER_EPOCH` to continue the same run toward 100.
Explicit resume paths must refer to that output directory's `last.pth`. To
relocate an existing run, copy the entire run directory so its best checkpoint,
history, and manifest remain available together.

The best checkpoint may precede uncertainty warmup. The standalone evaluator
reports that state and omits its untrained confidence diagnostics/safe variant;
evaluate a later checkpoint explicitly to study its trained uncertainty heads.

## Direct CLI and held-out comparison

Run training/evaluation on a compute node with the project environment active:

```bash
python3 scripts/train_prism.py --config research/configs/prism.json --list
python3 scripts/train_prism.py --config research/configs/prism.json --model prism_residual
python3 scripts/train_prism.py --config research/configs/prism.json --model all --resume auto

python3 scripts/train_prism.py --config research/configs/prism_smoke.json
```

The evaluator requires staged source images and an explicit manifest. For a
lossless dataset:

```bash
python3 scripts/evaluate_prism.py \
  --config research/configs/prism.json \
  --weights runs/prism_lossless/prism_residual/best.pth \
            runs/prism_lossless/prism_polar/best.pth \
            runs/prism_lossless/prism_cartesian_prob/best.pth \
  --manifest research/manifests/lossless_full.jsonl \
  --dataset-root data/research_lossless \
  --output-dir runs/prism_comparison --device cuda
```

All models receive identical crops/observations. The evaluator includes bilinear,
bicubic, and Lanczos; writes per-image chroma/RGB PSNR/SSIM, color and edge errors;
and reports per-condition paired improvements with bootstrap intervals.
Confidence ranking uses only model predictions, with errors used solely for
assessment. It rejects test images whose hashes were used in a checkpoint's
training or validation split. Keep `manifest.jsonl` beside copied checkpoints.

This evaluator covers synthetic subsampling. Real JPEG/H.264/HEVC/AV1 tests and
the existing historical-model benchmark remain separate workflows; no new codec
performance result is implied by this training suite.

### COCO 2014 held-out test

The user selected COCO 2014 test images as the final test set and bilinear
interpolation as the baseline. The archives live at
`/ocean/projects/cis260224p/shared/brios/data/coco/test2014.zip` and
`annotations/image_info_test2014.zip` within the same COCO directory.
The prepared manifest is `manifests/prism_coco_test2014.jsonl` there; its adjacent
`.meta.json` records archive/manifest SHA-256, counts, exclusions, and the training
manifests checked for overlap. All retained records have split `test`.

Verified September 5, 2026: 40,775 archive images, 40,499 eligible test images,
272 images below the crop size, four overlapping training/validation sources,
and no further duplicate-content exclusions. All image ZIP CRCs and metadata
dimensions passed. The 6,660,437,059-byte archive has SHA-256
`ead40c62230cb2cf70ff4c8b4c70abdc260a7556e77b3282621d06d8e2e35bdf`.
Preparation and bilinear-only evaluation regression tests passed. Full model
evaluation has not been run on this test set.

Testing follows the existing paired protocol: one centered 256×256 crop per
eligible image, the same twelve degradation conditions for each method, preserved
luminance, and chroma/RGB metrics with paired gains over bilinear. Images smaller
than the crop and repeated content are excluded from the evaluation manifest.
Training/validation overlap is excluded by COCO image ID or SHA-256. The evaluator
also checks content overlap against every supplied checkpoint's own manifest.

After choosing checkpoints using validation, submit from the repository:

```bash
sbatch --export=ALL scripts/bridges2/evaluate_prism_test2014.sbatch \
  /ocean/projects/cis260224p/shared/brios/checkpoints/prism/prism_coco/prism_residual/best.pth \
  /ocean/projects/cis260224p/shared/brios/checkpoints/prism/prism_coco/prism_polar/best.pth \
  /ocean/projects/cis260224p/shared/brios/checkpoints/prism/prism_coco/prism_polar_prob/best.pth \
  /ocean/projects/cis260224p/shared/brios/checkpoints/prism/prism_coco/prism_cartesian_prob/best.pth
```

This requests one L40S GPU for at most eight hours. Evaluation runtime for this
full set has not been measured. Results go to
`/ocean/projects/cis260224p/shared/brios/evaluations/prism_coco_test2014/<job-id>/`.
Downloading/preparing the test set does not submit this evaluation job.
