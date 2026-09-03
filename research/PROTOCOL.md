# Expanded neural chroma reconstruction protocol

This protocol replaces the undocumented 3,703-image historical evaluation with
a hash-addressed, paired experiment. It is designed so every aggregate value can
be traced to an image, degradation, method, and exact checkpoint.

## Data requirements

- Source images must be explicit RGB/RGBA 4:4:4 rasters in a lossless container
  (PNG, TIFF, BMP, or PPM). Both the decoded format and image mode are checked;
  a JPEG renamed with a lossless extension and grayscale/paletted inputs are
  rejected.
- Training, validation, and test paths must be physically separate. SHA-256
  digests are compared across splits to catch copied or renamed duplicates.
- Near-duplicate and scene-family filtering remains a dataset-curation
  responsibility; `source_group` should be used to audit burst frames and crops.
- The manifest and its metadata must be committed before the benchmark is run.

## Factorial degradation evaluation

The default benchmark includes centered, horizontally cosited, and fully
cosited chroma grids; box, triangle, Gaussian, and Lanczos-prefiltered
downsampling; and nearest, bilinear, bicubic, Lanczos, guided-filter, and joint
bilateral reconstruction. Every method receives the same low-resolution chroma
samples within a condition.

Actual codec tests use Pillow's local JPEG backend with explicit 4:2:0 sampling
at qualities 30, 50, 70, and 90. If a local FFmpeg provides the encoders, one
frame is round-tripped through H.264, HEVC, and AV1 as `yuv420p` at four CRF
settings each. Missing encoders are written as structured skips, never silently
omitted.

## Learned methods and ablations

The committed V5 and V6 checkpoints are benchmarked alongside classical
methods. The local training matrix also provides a three-layer, chroma-adapted
SRCNN baseline and a shallow NAF-style restoration baseline; the runner accepts
project-trained checkpoints and TorchScript adapters for other retrained
architectures. The preregistered one-factor-at-a-time matrix tests:

- luma guidance on/off;
- global bilinear residual versus direct chroma prediction;
- residual-block skips on/off;
- depth 4, 8, and 12;
- width 32, 64, and 96; and
- L1, MSE, Charbonnier, and L1-plus-gradient losses.

The SRCNN and NAF-style entries are architecture baselines, not part of the
one-factor attribution. They use the same inputs, manifest, degradation draws,
optimizer, seed, and checkpoint-selection rule as the V6-style base model.

All ablations use the same manifest, crops, degradation schedule, optimizer,
seed, and validation selection rule.

V7 extends this protocol without changing the V6 matrix. Its controlled first
milestone compares the unchanged V6 Cartesian model, V7 polar deterministic,
and V7 probabilistic models at width 64/depth 8. A separately reported
`v7_safe` row reuses the probabilistic checkpoint with inference-time gating;
the forward-consistency variant changes only its configured loss. V7 checkpoints
are selected on validation Cartesian chroma L1, never on the scientific test
split. Existing classical, V5/V6, TabPFN (where separately run), SRCNN, and
NAF-style controls must remain in the benchmark configuration.

V7 uncertainty reporting includes raw error/scale/kappa/confidence pixel maps,
amplitude interval coverage and width, magnitude-weighted circular phase error,
uncertainty/error Spearman correlation, and confidence risk-coverage at
10/25/50/75/100%. Correlation is evidence of ranking association only; neither
correlation nor nominal likelihood output is called calibration. Approximate
von Mises interval widths are identified as numerical CDF approximations.

Optional self-training is a distinct second-stage experiment. Its unlabeled
manifest is hash-checked against the held-out test split. The stage retains its
initial teacher, logs acceptance/confidence/disagreement, enforces supervised
examples in every update, caps the pseudo-loss weight, and requires uncertainty,
phase-concentration, forward-consistency, and center-siting-safe augmentation
consistency checks. It never overwrites the supervised checkpoint.

## Outputs

The benchmark writes:

- `per_image.jsonl`: every image × condition × method observation;
- `report.json`: mean, standard deviation, median, extrema, paired directional
  deltas, win rates, and bootstrap confidence intervals;
- `config.snapshot.json`: the exact run configuration;
- runtime mean/median/p95, parameter count/bytes, convolutional FLOPs, Python
  allocation peak, process-RSS delta, and accelerator allocation peak;
- actual encoded byte counts for codec conditions and a top-level profile table
  for each learned method and evaluated resolution;
- RGB and chroma PSNR/SSIM, mean and p95 CIEDE2000, chroma/edge/gradient MAE;
- difficult-boundary contact sheets and a machine-readable crop index; and
- Git revision, manifest digest, package versions, device, and FFmpeg
  capability provenance, including whether the worktree was dirty when the run
  began.

## Required execution order

1. Build and commit the lossless manifest.
2. Train the ablation matrix with `scripts/run_ablation_matrix.py`.
3. Add the generated ablation checkpoints to `learned_methods` in the benchmark
   configuration.
4. Run `scripts/research_benchmark.py`.
5. Commit the per-image data, report, configuration snapshot, and qualitative
   index before updating manuscript tables.

Historical and pilot numbers must remain visually separated from this protocol
until the full local corpus has been evaluated.
