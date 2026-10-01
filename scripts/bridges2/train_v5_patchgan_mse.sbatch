#!/usr/bin/env bash
# Original V5 generator and PatchGAN, trained with MSE on the Prism COCO split.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit through scripts/bridges2/submit_v5_patchgan_mse.sh" >&2
    exit 2
fi

project_root=${NCR_CODE_ROOT:-$SLURM_SUBMIT_DIR}
storage_root=${NCR_PROJECT_ROOT:-/ocean/projects/cis260224p/shared/$USER}
reference_run=${NCR_REFERENCE_RUN:-/jet/home/brios/checkpoints/prism/prism_coco_unet_mse100/prism_unet_mse}
output_dir=${NCR_OUTPUT_DIR:-/jet/home/brios/checkpoints/v5_patchgan_mse100}
archive=${NCR_DATASET_ARCHIVE:-$storage_root/data/coco/unlabeled2017.zip}
for required in "$project_root/scripts/train_v5_patchgan_mse.py" "$reference_run/manifest.jsonl" "$reference_run/config.snapshot.json" "$archive"; do
    if [[ ! -f "$required" ]]; then echo "Missing input: $required" >&2; exit 1; fi
done

if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module purge
module load pytorch/26.05-2.11-py3
export PYTHONPATH="$project_root/.cluster/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python3 -c 'import cv2, torch; assert torch.cuda.is_available(), "CUDA unavailable"'

stage_root=$(mktemp -d "${LOCAL:-/tmp}/v5-patchgan-mse-${SLURM_JOB_ID}.XXXXXX")
dataset_root="$stage_root/data"
mkdir -p "$dataset_root" "$output_dir"
echo "Staging COCO archive in $dataset_root"
unzip -q "$archive" -d "$dataset_root"
cd "$project_root"
args=(--reference-run "$reference_run" --dataset-root "$dataset_root"
      --output-dir "$output_dir" --batch-size "${NCR_BATCH_SIZE:-64}"
      --workers "${NCR_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}" --device cuda)
if [[ -f "$output_dir/last.pth" ]]; then args+=(--resume); fi
if [[ -n "${NCR_STOP_AFTER_EPOCH:-}" ]]; then
    args+=(--stop-after-epoch "$NCR_STOP_AFTER_EPOCH")
fi
echo "Starting V5 PatchGAN MSE job $SLURM_JOB_ID on $(hostname); output $output_dir"
nvidia-smi
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --cpu-bind=cores \
    python3 -u scripts/train_v5_patchgan_mse.py "${args[@]}"
