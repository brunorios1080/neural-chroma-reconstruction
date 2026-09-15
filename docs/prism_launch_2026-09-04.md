# Staged Prism launch — September 4, 2026

User authorized the staged smoke/core/review/conditional-expansion plan.

## Submitted

- Smoke job **45219528**: all twelve models sequentially on one L40S-48 GPU;
  production architecture sizes, batch 16, 256-pixel crops, 100 eligible COCO
  images, two epochs. Uncertainty and adversarial losses start immediately.
  Configuration: `research/configs/prism_gpu_smoke.json`. One-hour walltime cap.
- Core array **45219535**, indices 0–3: `prism_residual`, `prism_polar`,
  `prism_polar_prob`, `prism_cartesian_prob`, in that order. Each requests one
  L40S-48 GPU, with concurrency limited to two and eight hours per task.
  Configuration: `research/configs/prism_coco.json`, `NCR_STOP_AFTER_EPOCH=25`.
  The planned schedule remains 100 epochs. Complete eligible COCO dataset.
- Dependency: `afterok:45219528`, with `--kill-on-invalid-dep=yes`. No core task
  can start unless the entire twelve-model smoke process succeeds.
- Remaining eight full experiments have **not** been submitted.

Allocation balance at submission: 1,981 SU. This initial set requests at most
33 GPU-hours of walltime (1 + 4 × 8); actual runtime is not yet measured.
The eight-hour limit can interrupt a run before epoch 25. Completed epochs have
resumable checkpoints; no automatic repeated submissions are installed.

## Outputs and follow-up

Smoke outputs:
`/ocean/projects/cis260224p/shared/brios/checkpoints/prism/prism_gpu_smoke_smoke100/`

Core outputs:
`/ocean/projects/cis260224p/shared/brios/checkpoints/prism/prism_coco/`

Logs: `slurm_logs/prism-smoke-45219528.{out,err}` and
`slurm_logs/prism-45219535_<index>.{out,err}`.

```bash
squeue -u "$USER"
sacct -j 45219528,45219535 --format=JobID,JobName,State,Elapsed,ExitCode
/opt/packages/AI/pytorch_26.05-py3/bin/python3 scripts/review_prism.py \
  --output-root /ocean/projects/cis260224p/shared/brios/checkpoints/prism/prism_coco
```

At the replacement submission both stages were pending: smoke awaiting available
nodes, core gated on the smoke dependency.
Smoke success and scientific comparison have not yet been established.
Once all four reach 25, review validation curves, per-degradation gains, numerical
stability, uncertainty behavior, and measured compute before choosing extensions
or multi-seed replication. Do not rank models using the two-epoch smoke metrics.
No unattended scientific selection or further submission is installed.

## Corrected launch

The original smoke job **45217054** loaded 115,196 training images and 4,903
validation images instead of the intended 100-image subset. The submission
environment has `SBATCH_EXPORT=NONE`, which dropped the `NCR_MAX_IMAGES` override.
That job was cancelled after 17 minutes 29 seconds. Its partial output remains
under `prism_gpu_smoke/`, separate from the corrected `_smoke100/` output.

The GPU smoke config now contains `max_images: 100`; the batch launcher applies
it during preparation and rejects manifests outside the allowed 50–100 image
range before training. `NCR_MAX_IMAGES` remains an explicit override. Submission
now specifies `--export=ALL` to override the site's environment export setting.
Five launcher regression tests pass, including oversized-manifest rejection and
submission with `SBATCH_EXPORT=NONE`; shell syntax and whitespace checks pass.

The original core array **45217103** was held before cancelling the smoke, then
cancelled and replaced with **45219535**. The replacement explicitly receives
`NCR_STOP_AFTER_EPOCH=25` and `NCR_DEPENDENCY=afterok:45219528` through the corrected
submission wrapper. This preserves the original four models, two-task concurrency,
eight-hour caps, and full dataset. The reported balance remains 1,981 SU.

## Resume to epoch 25 — September 5, 2026

After reviewing the COCO 2014 comparison, the user authorized resuming the four
core models to the planned 25-epoch milestone. Array **45297762**, tasks 0–3,
maps to `prism_residual`, `prism_polar`, `prism_polar_prob`, and
`prism_cartesian_prob`, in that order. The first two resume after epoch 21;
the latter two resume after epoch 20, from each live run's `last.pth`.

All saved experiment settings match the current config. The launcher uses
`--resume auto` and exports `NCR_STOP_AFTER_EPOCH=25`; the planned 100-epoch
learning-rate horizon, optimizer state, and training data remain intact.
Each task requests one L40S GPU with a three-hour walltime cap and concurrency
two, for at most 12 GPU-hours across this submission. Prior epoch timings suggest
roughly 1.5–2 hours per task. The reported allocation balance at submission is
1,921 SU. Logs are `slurm_logs/prism-45297762_<index>.{out,err}`.

The completed test campaign uses separate frozen checkpoints. Its reported
results remain those of the earlier snapshots. This resume does not submit
another test-set evaluation or change the loss settings.
