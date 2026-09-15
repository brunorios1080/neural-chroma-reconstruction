# Paper-inspired V6 and Prism hybrid

The hybrid uses the independent scaled-color reconstruction as a starting image,
then trains an existing V6 or Prism network to correct its remaining chroma error.
It adds one learned scalar to the backbone. This is a new experiment; the prior
V6/Prism checkpoints and published comparison results retain their original meaning.

```
RGB source (encoder/training only)
  -> color transform, integer rounding, 4:2:0 sampling
  -> analytic interpolation and RGB decoding
  -> decoded RGB baseline + known sampling mask
  -> float YCrCb adapter -> V6 or Prism -> chroma correction
  -> zero-luma RGB correction, masked at retained sites, multiplied by tanh(gain)
  -> baseline + correction -> RGB8 clipping/rounding for evaluation
```

The network receives only the decoded baseline and the known sampling layout.
The original image is a training target, never a neural inference input. The
analytic encoder necessarily sees the source, as in the preceding comparison.
Scaled/lifting codes are not ordinary YCrCb values: feeding those directly to an
old checkpoint would misinterpret their range and meaning. The float adapter here
avoids another YCrCb quantization step; its neutral chroma is 128/255 for legacy V6
and 0.5 for Prism.

The correction changes chroma while preserving the baseline's floating luminance.
Its inverse uses the exact inverse coefficients of the adapter, and it is added
directly to the baseline RGB. Luminance equality is before final output clipping
and rounding. At point-sampled positions, the correction is forced to zero.
Those positions equal the original RGB under the lifting/decode-first variant;
under conventional or scaled-matrix reconstruction, their existing quantization
error is preserved. Centered box sampling has no exact retained-site constraint.

The guarded variant additionally preserves locally constant chroma. It measures
the range of R−G and B−G over a 5×5 neighborhood in the decoded baseline, and
suppresses correction when both ranges are at most 1e-6 in normalized units.
That tolerance only accommodates float32 cancellation and is far below one RGB8
level. This protects grayscale edges even when luminance changes. It uses no
target pixels, adds no learned parameters, and is saved as `flat_guard_radius=2`
in checkpoint metadata. Unguarded checkpoints retain `flat_guard_radius=0`.

`tanh(gain)` starts at zero. Therefore, every hybrid starts with exactly the
analytic baseline output, even with pretrained network weights. The first update
trains the gate; subsequent updates can train the backbone. This initialization
does not guarantee improvement after training or on unseen images. Epoch zero is
included in validation checkpoint selection, so a failed pilot can retain the
baseline. Prism uncertainty heads are frozen and are not calibrated or evaluated
by this RGB reconstruction experiment.

## What is implemented

- `chroma/li2026_hybrid.py`: analytic baseline preparation, V6/Prism wrapper,
  deterministic crops, RGB metrics, and portable checkpoint loading.
- `chroma/li2026_hybrid_training.py`: encoded-source hash verification, separation
  of dataset splits, training, validation selection, and epoch-boundary resume.
- `scripts/train_li2026_hybrid.py`: common runner for a named case or all cases.
- `research/configs/li2026_hybrid_pilot.json`: fixed four-case pilot.
- `research/configs/li2026_hybrid_guarded_pilot.json`: the same pilot with the
  constant-chroma guard enabled.
- `scripts/bridges2/li2026_hybrid_pilot.sbatch`: one V100, ten-minute hard cap;
  invariant/resume tests run before the pilot.

The checkpoint format is `li2026-hybrid-v1`. `best.pth` contains the architecture,
protocol, weights, selected epoch, and initialization provenance, and loads without
the original checkpoint file. `last.pth` also contains optimizer state, RNG state,
history, and the best state. Resume checks code, protocol, data selection, model,
and training settings. It reproduces completed epoch boundaries, not a partially
completed epoch. Configuration changes require a new run directory.

## Pilot protocol

The source is COCO **unlabeled2017**, using the existing Prism manifest's original
train/validation assignments. The pilot chooses 256 training images and 64
validation images deterministically by seed and ID, before examining quality.
Every case uses those same images. Encoded-image SHA-256 values are checked before
training. The archive is read directly without unpacking the full dataset.

Training uses 128×128 random crops, rotations and flips applied before simulated
encoding, batch size 8, four epochs, RGB MSE, Adam, backbone LR 0.00003 and gate LR
0.003. Validation uses one fixed center crop per image. There is no scheduler,
mixed precision, or TF32. Augmentation and order are deterministic by epoch and
image ID. This is a feasibility pilot, not a full-data convergence study.

The four cases are V6 with conventional bilinear, V6 with scaled-matrix bilinear,
V6 with lifting/decode-first bicubic, and Prism residual with lifting/decode-first
bicubic. All use point sampling. Each learned output is paired with its own
analytic baseline. Differences between analytic variants also include changes in
transform or interpolation, so only each within-case baseline/hybrid difference
isolates the learned correction.

Selection maximizes mean per-image RGB CPSNR after clipping and rounding to RGB8.
RGB MAE, MSE and image wins are also recorded. To keep checkpoint/JSON metrics
finite on perfectly reproduced crops, MSE is floored at 1e-12 for CPSNR only
(168.1308 dB ceiling); reported MSE and MAE remain unmodified. Training uses smooth,
unclipped floating RGB MSE. All per-image validation results are saved for review.

UHD, Kodak and COCO test2014 are not used for training or checkpoint selection in
this experiment. V6's original pretraining image identities remain unknown, so
we can certify this fine-tuning split, not a historical V6 pretraining holdout.
UHD/Kodak results were already examined in the preceding exploratory comparison;
an additional untouched dataset would strengthen a final confirmatory study.

## Running and loading

Run training on a compute node with the project PyTorch environment:

```bash
python3 scripts/train_li2026_hybrid.py \
  --config research/configs/li2026_hybrid_pilot.json --output /path/to/new/run
# Continue the same run after an interruption:
python3 scripts/train_li2026_hybrid.py \
  --config research/configs/li2026_hybrid_pilot.json --output /path/to/existing/run \
  --case v6_decode_first_bc --resume
```

The Slurm script requires `HYBRID_OUTPUT`, a new output directory. Submit from the
repository or a complete frozen code snapshot; create `slurm_logs` first. The
pilot submission used for this experiment freezes its source files and hashes.

For inference, `load_hybrid(path, device)` returns `(model, checkpoint)`. Call
`model.eval()` and, inside `torch.no_grad()`, pass float RGB `[B,3,H,W]` in `[0,1]`
and a Boolean `[B,1,H,W]` retained-site mask. Inputs must come from the decoder
protocol saved in the checkpoint. Clip/round the returned RGB only when exporting
or measuring RGB8 output. `prepare_baseline(source_rgb_uint8, model.protocol)`
simulates the corresponding encoder/decoder for controlled evaluations.

## Relationship to the paper and next experiment

The [September 6 pilot results](li2026_hybrid_pilot_2026-09-06.md) include both the
initial unguarded run and the subsequent constant-chroma guard. The latter gave
positive validation gains in all four cases. Full benchmark evaluation has not
been run for these new hybrid checkpoints.

The source is [Li, Zhang and Huang, IET Image Processing (2026)](https://doi.org/10.1049/ipr2.70338).
Our analytic code is an independent implementation. The `scaled_decode_first`
variant is our explicit decoding interpretation, not author-supplied code.
Exact reproduction of the paper's image tables remains unverified; see the
[previous protocol and comparison](li2026_comparison_2026-09-05.md).

Any pilot gain is over that specified local baseline. It is not evidence that
the hybrid beats the authors' actual implementation, generalizes to UHD, or
improves equal-bitrate compression. Scaled integer channels can require a wider
storage range, which still needs a rate comparison.

If the pilot is promising, the next experiment is longer matched COCO training,
followed by a fixed-checkpoint evaluation on all UHD and Kodak images. Include
point and centered-box conditions, all four Prism backbones, multiple training
seeds, paired image differences, inference time/memory, and encoded size. Fix
those choices before examining the final test results.
