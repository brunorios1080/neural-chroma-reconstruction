# Neural Chroma Reconstruction

A research project by **Bruno Rios**, University of Texas Rio Grande Valley.

This project reconstructs full-resolution chroma from images degraded with a
synthetic 4:2:0 pipeline. Luma is retained at full resolution; Cr and Cb are
downsampled by two in each dimension and bilinearly restored before entering a
model.

## Models

The new **Prism** family provides twelve independently named, matched training
experiments derived from the model review, including residual/NAF baselines,
polar and Cartesian uncertainty variants, degradation conditioning, measurement
consistency, and chroma-only U-Nets. See [Prism training and evaluation](docs/prism_training.md)
for the model map, smoke checks, Bridges-2 submission, monitoring, and resume commands.

Saved experiment weights and their run metadata are under [models/experiments](models/experiments/README.md).
The train, validation, and test image IDs are under [research/manifests/ids](research/manifests/ids/README.md);
the COCO image datasets are stored separately.

| Model | Parameters | Design | Training objective |
| --- | ---: | --- | --- |
| V5 | 1,925,667 generator + 694,241 discriminator | U-Net and PatchGAN that reconstruct full YCrCb | adversarial loss + 10x full-image L1 |
| V6 | 593,794 | eight-block residual CNN that passes Y through unchanged and predicts Cr/Cb corrections | chroma-only L1 |
| V7 | 595,525 | V6 trunk with residual polar magnitude/circular phase mean and separate uncertainty maps | Cartesian L1 + Laplace magnitude NLL + magnitude-weighted von Mises phase NLL |

V5 is the perceptual/hallucination experiment. Because it regenerates all three
channels and uses an adversarial objective, it can trade pixel accuracy for
plausible detail. V6 is the conservative refiner and is the recommended baseline
for objective fidelity.

### V7: probabilistic polar chroma reconstruction

V7 is an additional experiment; it does not replace V6. V6 predicts a
deterministic Cartesian residual, `C_hat = C_bilinear + R(X)`. V7 keeps V6's
3-to-64 stem and eight residual blocks, but changes the output parameterization.
For repository-ordered chroma `[Cr, Cb]` and explicit neutral point `c0`,

```text
u = Cr - c0                     v = Cb - c0
A = sqrt(u^2 + v^2)             phi = atan2(v, u)
Cr = c0 + A cos(phi)             Cb = c0 + A sin(phi)
```

Its five-map head predicts an amplitude residual, a normalized cosine/sine phase
residual, a Laplace amplitude scale `b_A`, and a von Mises phase concentration
`kappa`. The supervised objective is

```text
L = lambda_cart L1(C_hat, C)
  + lambda_amp [|A-A_hat|/b_A + log(2 b_A)]
  + lambda_phase w_phi [-kappa cos(phi-phi_hat) + log(2 pi I0(kappa))]
  + lambda_forward L1(D(C_hat), C_low),

w_phi = clamp(A / phase_reference_amplitude, 0, 1).
```

The forward term is optional and uses the observation's exact siting and
prefilter. V7 also exposes 50/80/90/95% amplitude intervals, kappa, circular
variance, approximate circular von Mises interval half-widths, and two inference
modes: `mean`, and `safe`, which continuously backs residuals toward bilinear
according to configured uncertainty-to-confidence functions. Safe gating is
inference-only and cannot suppress training residuals.

The publication pipeline uses analytical full-range BT.601 with `c0=0.5`.
Legacy OpenCV tensors instead use the uint8 neutral code `128/255`; V7 records and
strictly validates this setting in checkpoints rather than guessing it. See
[`docs/v7_design.md`](docs/v7_design.md) for the full repository audit and
numerical policy.

Supervised training (not run automatically):

```bash
python scripts/train_v7.py --config research/configs/v7.json
```

### Bridges-2 cluster training

The Bridges-2 launch path uses PSC's maintained PyTorch module, adds only
headless OpenCV in a repository-local dependency directory, reads the packed COCO
ZIP from `/ocean`, and extracts it only onto compute-node local storage. It does
not create 123,403 JPEG files in `$HOME` or `/ocean`. Checkpoints and model outputs
default to `/ocean/projects/cis260224p/shared/$USER/checkpoints/v7/`.

Run setup once on the login node:

```bash
./scripts/bridges2/setup.sh
```

Before the first submission, put the packed archive and metadata in project storage:

```text
/ocean/projects/cis260224p/shared/$USER/data/coco/unlabeled2017.zip
/ocean/projects/cis260224p/shared/$USER/data/coco/annotations/image_info_unlabeled2017.json
```

First submit a 100-image, one-epoch smoke run capped at 30 minutes:

```bash
NCR_MAX_IMAGES=100 NCR_EPOCHS=1 NCR_WALLTIME=00:30:00 \
  ./scripts/bridges2/submit_v7.sh research/configs/v7_coco.json
```

After the smoke run succeeds, submit the full V7 run:

```bash
./scripts/bridges2/submit_v7.sh research/configs/v7_coco.json
```

The tracked COCO V7 configuration trains for 100 epochs with batch 64. Smoke checkpoints are
automatically isolated in a `_smokeN` output directory. Production checkpoints
are written after every epoch. A later submission automatically resumes from
the output directory's `last.pth`; `NCR_RESUME` can select another checkpoint.

The Bridges-2 COCO split uses 96% training and 4% validation. Finite-value checks
remain enabled, and validation records Cartesian loss plus chroma/RGB PSNR and
SSIM. Batch 64 has been verified on a 48 GB L40S.

Submit multiple model configurations as independent one-GPU jobs by listing
each config. Their output directories must be distinct:

```bash
./scripts/bridges2/submit_v7.sh \
  research/configs/v7_coco.json \
  path/to/second_v7_config.json
```

Defaults target one 48 GB L40S in the `GPU-shared` partition under allocation
`cis260224p`. Override resource choices with `NCR_GPU`, `NCR_CPUS`,
`NCR_MEMORY`, `NCR_WALLTIME`, `NCR_ACCOUNT`, or `NCR_PARTITION`. Training
overrides include `NCR_BATCH_SIZE`, `NCR_EPOCHS`, `NCR_WORKERS`, and
`NCR_RESUME`. `NCR_MAX_IMAGES` restricts the manifest for smoke testing;
`NCR_OUTPUT_DIR`, `NCR_DATASET_ARCHIVE`, `NCR_COCO_METADATA`, and
`NCR_PROJECT_ROOT` override the `/ocean` storage layout. Slurm logs are written
under `slurm_logs/`.

The COCO configuration is useful for synthetic-degradation training but is not
a lossless publication dataset; keep those evidence tiers separate.

Run the controlled polar deterministic / probabilistic / forward matrix:

```bash
python scripts/run_v7_ablations.py --config research/configs/v7_ablations.json
```

Evaluate V7 mean and safe outputs, render uncertainty maps, and write
`per_image.jsonl`, pixel `.npz` maps, interval coverage, uncertainty/error
correlations, and risk-coverage curves:

```bash
python scripts/evaluate_v7.py --config research/configs/v7_evaluation.json
```

To add the two modes to the existing full publication benchmark without
removing any existing baseline, append these entries to a copy of its
`learned_methods` list:

```json
[
  {"name":"v7_mean","type":"v7","mode":"mean","weights":"research/checkpoints/v7/v7_probabilistic/best.pth"},
  {"name":"v7_safe","type":"v7","mode":"safe","weights":"research/checkpoints/v7/v7_probabilistic/best.pth"}
]
```

The optional EMA teacher/student stage is disabled in its checked-in config and
must be explicitly enabled. Thresholds are starting hypotheses, not tuned
optima:

```bash
python scripts/build_research_manifest.py \
  --dataset-root data/unlabeled \
  --unlabeled data/unlabeled \
  --output research/manifests/unlabeled.jsonl

python scripts/self_train_v7.py \
  --config research/configs/v7_self_train.json --enable
```

Software-only smoke and hard behavioral fixtures are separate from scientific
evaluation:

```bash
python scripts/v7_smoke.py
python scripts/v7_hypothesis_fixtures.py \
  --weights research/checkpoints/v7/v7_probabilistic/best.pth \
  --output research/reports/v7/hypothesis_fixtures.json
```

The fixture includes indistinguishable same-luma/different-chroma observations,
neutral chroma, the ±pi hue boundary, and a sharp chroma edge. It validates
software behavior only and cannot establish recovery or calibration.

Print the model sizes with:

```bash
python scripts/model_summary.py
```

## Historical result summary

These values were recorded by the original experiments on 3,703 validation
images. The original dataset manifest is not committed, so treat them as archived
results rather than a reproducible benchmark. The repository's evaluator can be
used to produce new JSON reports.

| Model | Metric | Bilinear | Model | Delta |
| --- | --- | ---: | ---: | ---: |
| V5 | RGB PSNR | 40.42 dB | 37.11 dB | **-3.31 dB** |
| V5 | RGB SSIM | 0.9816 | 0.9761 | **-0.0055** |
| V5 | Chroma PSNR | 42.88 dB | 40.98 dB | **-1.90 dB** |
| V5 | Chroma SSIM | 0.9530 | 0.9536 | **+0.0006** |
| V6 | RGB PSNR | 40.42 dB | 44.04 dB | **+3.62 dB** |
| V6 | Chroma PSNR | 42.88 dB | 46.76 dB | **+3.88 dB** |

On the historical video experiment, V6 improved PSNR from 57.54 dB to 58.55 dB
and SSIM from 0.9985 to 0.9988. V5 did not beat bilinear interpolation on PSNR;
its only reported average improvement was a small chroma-SSIM increase.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For CUDA training, install the PyTorch build appropriate for your CUDA runtime
before installing the remaining requirements.

## Prepare a dataset

Training scans directories recursively, so preparation is optional. The helper
below validates images, rejects files smaller than the requested crop, and copies
the usable files while preserving their directory layout:

```bash
python scripts/prepare_dataset.py \
  --input data/raw \
  --output data/processed \
  --min-size 256
```

## Train

The V5 and V6 legacy entry points use the same deterministic split, validation loop, sample writer,
history log, and resumable checkpoint format.

```bash
# V5 GAN
./scripts/run_v5.sh --src data/processed --epochs 30 --batch-size 4

# V6 residual refiner
./scripts/run_v6.sh --src data/processed --epochs 30 --batch-size 16
```

The equivalent unified interface is:

```bash
python scripts/train.py --model v6 --src data/processed
```

Useful options include `--crop`, `--workers`, `--val-fraction`, `--seed`,
`--device`, `--output-dir`, and `--no-amp`. V5 crops must be divisible by eight
and at least 64 pixels for its discriminator.

Each run writes:

- `last.pth`, `best.pth`, and per-epoch checkpoints;
- optimizer state, configuration, epoch, and best validation loss;
- `history.jsonl` with training and validation metrics;
- correctly color-converted `bilinear | target | model` sample images.

Resume a current checkpoint with:

```bash
./scripts/run_v6.sh \
  --src data/processed \
  --epochs 60 \
  --resume runs/v6/checkpoints/last.pth
```

The committed legacy checkpoints remain supported. Their weights can be used for
inference and evaluation, but they lack optimizer state and therefore cannot
resume exactly from the original training run.

### V5.1 continuation on Bridges-2

V5.1 preserves `models/version5/epoch_010.pth` and continues its generator and
discriminator weights from epoch 11 through epoch 100. Because the historical
checkpoint has no optimizer state, V5.1 recreates both documented Adam optimizers
at `2e-4`; it is a weight continuation rather than an exact optimizer continuation.
The cluster run uses batch 64 after H100 throughput benchmarks, while retaining
the original architecture, crop, loss, learning rate, and full-precision arithmetic.
New checkpoints are written separately under
`/ocean/projects/cis260224p/shared/$USER/checkpoints/v5.1/`.

Run a 100-image epoch-11 smoke continuation:

```bash
NCR_GPU=h100-80 NCR_MAX_IMAGES=100 NCR_EPOCHS=11 NCR_WALLTIME=00:30:00 \
  ./scripts/bridges2/submit_v5_1.sh
```

Submit the full continuation through epoch 100:

```bash
./scripts/bridges2/submit_v5_1.sh
```

## Inference

```bash
python scripts/inference.py \
  --model v6 \
  --weights models/version6/res_epoch_030.pth \
  --image sample.jpg \
  --output results/sample_v6.png \
  --comparison results/sample_v6_comparison.png
```

The comparison image contains `bilinear | model`. V5 inputs are automatically
padded and cropped back when their dimensions are not divisible by eight.

## Evaluate

The evaluator compares the AI output with the bilinear input using RGB and
chroma-only PSNR/SSIM:

```bash
python scripts/evaluate.py \
  --model v6 \
  --weights models/version6/res_epoch_030.pth \
  --src data/processed \
  --crop 256 \
  --json results/v6_metrics.json
```

Use `--crop 0` to evaluate complete images. The report records mean, minimum, and
maximum values and lists skipped images.

## Publication benchmark

The expanded local research protocol uses a SHA-256-addressed manifest of
explicit lossless RGB/RGBA 4:4:4 files and rejects byte-identical images across
train, validation, and test splits. It evaluates multiple chroma sitings and
prefilters, actual 4:2:0 JPEG output, optional local H.264/HEVC/AV1 round trips,
six classical methods, V5/V6, chroma-adapted SRCNN and NAF-style learned
baselines, and the full V6 ablation matrix.

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

Each run emits per-image JSONL, aggregate and paired bootstrap statistics,
configuration and provenance snapshots, RGB/chroma PSNR and SSIM, CIEDE2000,
edge errors, runtime, convolutional FLOPs, model and peak-memory measurements,
and difficult-color-boundary contact sheets. See
[`research/PROTOCOL.md`](research/PROTOCOL.md) for the preregistered conditions
and evidence rules.

The repository also commits a 12-image procedural fixture audit. It verifies the
entire local pipeline and produces 1,024 method-image-condition records across
all 28 configured groups, including real H.264, HEVC, and AV1 round trips at all
preregistered settings, but its generated
patterns are deliberately not presented as publication evidence. A scientific
large-scale rerun still requires a separately supplied local lossless corpus;
video-codec conditions additionally require an FFmpeg build with the requested
encoders.

## TabPFN V3 chroma imputation experiment

The optional hosted TabPFN experiment treats pixels as tabular rows. It uses
full-resolution luma and spatial features with the observed 4:2:0 Cr/Cb samples,
then fits one V3 regressor per chroma channel. Install `tabpfn-client`, configure
`TABPFN_TOKEN`, and run:

```bash
python scripts/tabpfn_chroma_impute.py \
  --image data/raw/coco/val2017/000000000139.jpg \
  --crop 64 \
  --output-dir results/tabpfn_v3_chroma
```

The output includes the original, bilinear reconstruction, TabPFN reconstruction,
a side-by-side comparison, and a JSON report with RGB/chroma PSNR and SSIM plus
chroma edge and gradient errors. API inference sends the crop's derived feature
table and low-resolution chroma targets to Prior Labs. This is an image-adaptive
imputation experiment, not a replacement training pipeline or a representative
dataset benchmark.

Run a seeded paired benchmark against bilinear, bicubic, V5, and V6 with:

```bash
python scripts/benchmark_coco_tabpfn.py \
  --src data/raw/coco/val2017 \
  --images 20 \
  --crop 64 \
  --seed 2026 \
  --output-dir results/coco_tabpfn_benchmark
```

Each completed crop is cached before the next API request, so rerunning the same
command resumes without consuming quota again. Pass `--overwrite` only when the
hosted predictions should be recomputed. The benchmark writes per-image metrics
and images, aggregate statistics, directional paired deltas, win rates, 95%
bootstrap confidence intervals, and a contact sheet. See the
[20-crop pilot summary](notes/tabpfn_v3_coco_pilot.md) for the first seeded run.

## Tests

```bash
./scripts/test.sh
```

The tests cover model parameter counts and output contracts, deterministic data
splitting, 4:2:0 simulation, metrics, arbitrary-size V5 inference, checkpoint
round trips, loading both committed legacy checkpoints, and complete one-epoch
training runs for both models.

## Scope and limitations

The historical V6 checkpoint was trained on one idealized degradation, and the
original 3,703-image dataset manifest is unavailable. The publication protocol
now measures cross-siting/filter robustness and actual codec output, but those
new results must not be claimed until the complete local lossless benchmark is
run and its per-image artifacts are committed. Single-frame codec tests also do
not establish temporal stability.

## License

MIT License © 2025 Bruno Rios
