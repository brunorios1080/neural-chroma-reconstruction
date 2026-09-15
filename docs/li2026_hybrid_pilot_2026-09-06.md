# V6/Prism analytic-neural pilot, September 6, 2026

The implemented hybrid improved each analytic baseline in a small COCO validation
pilot after adding a guard that preserves locally constant chroma. The strongest
result was Prism residual with the independent lifting/decode-first bicubic
baseline: **+1.2602 dB**, with lower RGB error on **61/64 validation crops**.

This is validation evidence from an adaptive development experiment, not a final
test result or a comparison against executable author code. No new UHD, Kodak or
COCO test2014 evaluation was run. See the [implementation and protocol](li2026_hybrid.md).

## Matched guarded pilot

All cases used the same 256 COCO unlabeled2017 training images, 64 validation
images, 128×128 crops, seed 20260906, four epochs and optimizer settings. Splits
came from the existing Prism manifest. Encoded-source hashes were verified. V6
initialized from the frozen epoch-30 checkpoint; Prism residual initialized from
the frozen epoch-21 checkpoint used in the preceding comparison, before the later
resumed training. V6's historical training image identities remain unknown.

| Backbone and analytic baseline | Baseline CPSNR | Hybrid CPSNR | Gain | Improved crops |
|---|---:|---:|---:|---:|
| V6, conventional bilinear | 39.4231 | 40.5760 | +1.1529 dB | 60/64 |
| V6, scaled-matrix bilinear | 43.1477 | 44.3473 | +1.1996 dB | 60/64 |
| V6, lifting/decode-first bicubic | 43.3382 | 44.4124 | +1.0741 dB | 59/64 |
| Prism residual, lifting/decode-first bicubic | 43.3382 | 44.5985 | +1.2602 dB | 61/64 |

Epoch four was selected in each guarded case. Every comparison uses the same
encoded observation and its own analytic baseline. Gains between different rows
also reflect different transforms/interpolators; the within-row gain isolates
the trained correction. Higher CPSNR means lower squared RGB reconstruction error.

The network adds a gated chroma residual to the already decoded RGB. Retained
point samples are protected. In this guarded run, constant chroma within a 5×5
neighborhood is also protected using only R−G and B−G from the decoded input.
The wrapper adds one learned scalar; the V6 and Prism residual hybrids each have
593,795 parameters. This does not include the analytic preprocessing's runtime
or memory cost, which has not yet been benchmarked.

## What the initial run revealed

The first run used the same setup without the constant-chroma guard. V6 with
conventional bilinear gained 1.1502 dB. The three scaled cases improved most crops
and reduced mean RGB MSE, but their mean CPSNR fell after introducing tiny errors
into two crops that the analytic baseline reconstructed perfectly. Consequently,
the validation selector correctly retained epoch zero for all three scaled cases.
Their last-epoch mean CPSNR changes were −1.5280 dB (V6 scaled matrix), −1.6672 dB
(V6 decode-first), and −1.4838 dB (Prism decode-first).

Those two validation IDs were `validation/000000473760` and
`validation/000000226990`. The guarded follow-up kept both exactly reproduced in
the scaled cases. This guard was designed after inspecting validation failures;
the follow-up is explicitly adaptive, and both runs are retained.

All reported CPSNR values use the same MSE floor of 1e-12 for perfect crops,
equivalent to a 168.1308 dB ceiling. The two perfect crops raise the absolute mean
CPSNR in scaled cases. Their contribution cancels in the guarded within-case
gain because both outputs remain perfect. Do not compare these crop-level
absolute means directly to full-image paper or UHD scores. RGB MSE and MAE are
recorded without a floor, and per-image results are available for inspection.

## Validation and artifacts

The initial job passed ten tests; the guarded job passed twelve. They cover exact
zero-gain baseline identity, retained-site and floating-luminance preservation,
constant-chroma preservation, finite gradients in V6 and all four frozen Prism
backbones, deterministic directory/ZIP crops, disjoint source verification,
portable checkpoint loading, and exact epoch-boundary resume on a small CPU fixture.
Only V6 and Prism residual were fine-tuned in this pilot; the other three Prism
backbones passed compatibility/gradient checks.

- Initial job: `45403007`, completed successfully in 87 seconds.
- Guarded job: `45403423`, completed successfully in 69 seconds.
- Total: 156 seconds on one V100, approximately **0.0433 SU**.
- Allocation balance reported afterward: **1,903 SU**. Neither job remains active.

Initial artifacts:
`/ocean/projects/cis260224p/shared/brios/checkpoints/li2026_hybrid/pilot_20260906/`

Guarded artifacts:
`/ocean/projects/cis260224p/shared/brios/checkpoints/li2026_hybrid/guarded_pilot_20260906/`

Each directory contains frozen code and SHA-256 hashes, job logs/provenance, and
four runs. Each run has `best.pth`, resumable `last.pth`, selected source manifest,
initialization provenance, epoch history, a report and per-image validation rows.
`scripts/report_li2026_hybrid.py` recomputes the summaries, checks data coverage,
matched selections and code hashes, and records final checkpoint hashes.

The initial run's independent artifact audit passed. The guarded run's training,
source verification and twelve preflight tests passed, and its saved summaries
were read successfully. A subsequent independent artifact audit could not finish:
shared-storage reads stalled on saved provenance and then a frozen source file.
That extra audit was stopped; it should be rerun when storage reads recover. The
guarded numbers above are the completed trainer's validation summaries.

## Next step

Train the guarded variants on the larger COCO training split with multiple seeds
and a fixed validation protocol, then freeze checkpoints for full-image testing.
Include matched interpolation controls, point and centered-box sampling, and the
remaining Prism backbones. Measure runtime, memory and encoded size alongside
quality. Add an untouched confirmatory dataset because UHD/Kodak were already
examined during the preceding exploration. These pilot results do not yet support
a claim of superiority to the published method or equal-bitrate compression.
