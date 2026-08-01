# Neural Chroma Reconstruction

A research project by **Bruno Rios**, University of Texas Rio Grande Valley.

This project reconstructs full-resolution chroma from images degraded with a
synthetic 4:2:0 pipeline. Luma is retained at full resolution; Cr and Cb are
downsampled by two in each dimension and bilinearly restored before entering a
model.

## Models

| Model | Parameters | Design | Training objective |
| --- | ---: | --- | --- |
| V5 | 1,925,667 generator + 694,241 discriminator | U-Net and PatchGAN that reconstruct full YCrCb | adversarial loss + 10x full-image L1 |
| V6 | 593,794 | eight-block residual CNN that passes Y through unchanged and predicts Cr/Cb corrections | chroma-only L1 |

V5 is the perceptual/hallucination experiment. Because it regenerates all three
channels and uses an adversarial objective, it can trade pixel accuracy for
plausible detail. V6 is the conservative refiner and is the recommended baseline
for objective fidelity.

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

Both versions use the same deterministic split, validation loop, sample writer,
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

## Tests

```bash
./scripts/test.sh
```

The tests cover model parameter counts and output contracts, deterministic data
splitting, 4:2:0 simulation, metrics, arbitrary-size V5 inference, checkpoint
round trips, loading both committed legacy checkpoints, and complete one-epoch
training runs for both models.

## Scope and limitations

The training degradation represents one idealized chroma filtering and siting
choice. It does not simulate codec quantization, ringing, bitrate changes, native
YUV decoding, or temporal behavior. Results should be validated on actual encoded
4:2:0 material before drawing codec-level conclusions.

## License

MIT License © 2025 Bruno Rios
