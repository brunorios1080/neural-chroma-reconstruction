# COCO 2014 test: all available model variants

The user authorized testing all available models against bilinear interpolation.
Campaign directory:
`/ocean/projects/cis260224p/shared/brios/evaluations/test2014_all_20260905/`.

## Jobs

- GPU preflight **45266325**: all 21 frozen models, 16 test images × 12 degradation
  conditions, one L40S GPU, 20-minute cap.
- Full array **45267224**, tasks 0–4, concurrency two: one L40S per task and an
  eight-hour cap per task, at most 40 GPU-hours across the full array.
- Full-array dependency: `afterok:45266325`, with `--kill-on-invalid-dep=yes`.
  Preflight passed all 21 models on 192 observations with finite metrics; its
  evaluation loop took 20.56 seconds excluding staging/loading. The full array
  is now waiting for Priority. Full-set results are not yet complete.
- Existing Prism training jobs were left running. The comparison uses copied
  checkpoints and copied evaluation code, not live changing checkpoint paths.

## Selected checkpoints

| Group | Models | Checkpoints |
| --- | --- | --- |
| 0 | Prism residual, polar, polar-probabilistic, Cartesian-probabilistic | Full-dataset validation-best checkpoints; selected epochs 21, 21, 5, 12 respectively |
| 1 | Prism MSE, edge, NAF, conditioned | Two-epoch, 100-image smoke checkpoints |
| 2 | Prism forward, polar-forward, U-Net, GAN | Two-epoch, 100-image smoke checkpoints |
| 3 | V5, V5.1, V6, V7 | Epochs 10, 13, 30, 100 respectively |
| 4 | Ablation base, direct, NAF-style, no-luma, SRCNN | One-epoch fixture smoke checkpoints |

There are 21 distinct variants. A main validation-best checkpoint represents each
variant when available; duplicate smoke/benchmark runs and intermediate epochs of
the same model are not separate candidates. Historical V5 and V6 use their sole
available committed checkpoints. V5.1 is a named retraining run of the V5 model
architecture. `campaign.json` records original paths, frozen paths, SHA-256,
epoch, parameter count, training status, and overlap-audit status for every model.
Smoke-only variants are diagnostic comparisons, not fully trained contenders.
The polar-probabilistic validation-best checkpoint is still from epoch five;
testing it does not measure the eventual fully trained uncertainty head.

## Matched test protocol

- 40,498 images: the prepared 40,499-image test manifest minus one additional
  overlap found when excluding the entire known COCO training source corpus
  (including its original reserved split) and ablation fixtures.
- Every image supplies a centered 256×256 crop and all twelve configured
  synthetic 4:2:0 siting/filter conditions: 485,976 observations per method.
- The same bilinear-upsampled observation feeds every model. Bilinear is the
  comparison reference. Mean model outputs are tested, with final [0,1] clipping.
- Per-image chroma/RGB PSNR and SSIM, chroma/full/luma L1, and out-of-range fraction
  use the existing `chroma.prism_metrics.quality_batch` implementation. This run
  does not compute CIEDE2000, risk-coverage curves, safe-output variants, or
  bootstrap confidence intervals from the slower comprehensive evaluator.
- Results report means, paired mean improvements, and win rates overall and
  per degradation. Training status accompanies each result.
- Known source overlap is excluded by image ID and content hash. Original
  training manifests are unavailable for historical-model checkpoints; those
  entries explicitly retain that limitation even though known COCO sources were
  excluded from their shared test set.

## Results and recovery

Each group writes `groups/<index>/metrics.npy`, `progress.json`, and, when
complete, `report.json`. The compact metric array stores one row per observation,
one column per method, and eight metric values. Row ordering is manifest image
order, then configured degradation order. Progress is committed every 16 batches;
an interrupted group resumes from its last committed position using the same
campaign, group, batch size, and output directory.

The last finishing group collects all reports into `comparison.json` and
`comparison.csv` in the campaign root. Collection requires all 21 models and
complete reports from the matching campaign. Logs are
`slurm_logs/coco2014-check-45266325.{out,err}` and
`slurm_logs/coco2014-all-45267224_<index>.{out,err}`.

```bash
squeue -u brios
sacct -j 45266325,45267224 --format=JobID,JobName,State,Elapsed,ExitCode
```

Validation before submission: all 21 checkpoint loaders succeeded; fixture tests
passed for identical-model/bilinear equality, interrupted metric-array recovery,
paired gain direction, condition indexing, and complete matching report
collection. Shell syntax and whitespace checks passed. The reported allocation
balance at launch was 1,947 SU. No automatic repeated submissions are installed.

The first queued full array, **45267118**, was cancelled before execution and
replaced to explicitly bind its evaluation step to all eight allocated CPUs.
The preflight exposed a one-core binding warning; the replacement full launcher
passes `--cpus-per-task` to `srun`.
