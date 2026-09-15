#!/usr/bin/env bash
#SBATCH --account=cis260224p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --time=1-00:00:00
#SBATCH --job-name=ncr-v7
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch; do not run training on a login node." >&2
    exit 2
fi
if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 CONFIG.json" >&2
    exit 2
fi

project_root=${NCR_CODE_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$project_root" || ! -f "$project_root/scripts/train_v7.py" ]]; then
    echo "Cannot locate the neural-chroma-reconstruction code checkout." >&2
    echo "Submit with scripts/bridges2/submit_v7.sh or set NCR_CODE_ROOT." >&2
    exit 1
fi
config=$1
if [[ "$config" != /* ]]; then
    config="$project_root/$config"
fi
storage_root=${NCR_PROJECT_ROOT:-/ocean/projects/cis260224p/shared/$USER}
archive=${NCR_DATASET_ARCHIVE:-$storage_root/data/coco/unlabeled2017.zip}
metadata=${NCR_COCO_METADATA:-$storage_root/data/coco/annotations/image_info_unlabeled2017.json}
config_stem=$(basename "$config" .json)
config_stem=$(printf '%s' "$config_stem" | tr -c '[:alnum:]_-' '-')
default_output_dir="$storage_root/checkpoints/v7/$config_stem"
if [[ -n "${NCR_MAX_IMAGES:-}" ]]; then
    default_output_dir="${default_output_dir}_smoke${NCR_MAX_IMAGES}"
fi
output_dir=${NCR_OUTPUT_DIR:-$default_output_dir}
resume=${NCR_RESUME:-}
if [[ -z "$resume" && -f "$output_dir/last.pth" ]]; then
    resume="$output_dir/last.pth"
fi

if [[ ! -f "$config" ]]; then
    echo "Missing V7 config: $config" >&2
    exit 1
fi
if [[ ! -f "$archive" ]]; then
    echo "Missing COCO archive: $archive" >&2
    exit 1
fi

if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
fi
module purge
module load pytorch/26.05-2.11-py3

export PYTHONPATH="$project_root/.cluster/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
export HF_HOME="$storage_root/cache/huggingface"
export TORCH_HOME="$storage_root/cache/torch"

python3 -c 'import cv2, torch' || {
    echo "Cluster dependencies are missing; run scripts/bridges2/setup.sh first." >&2
    exit 1
}
python3 -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' || {
    echo "CUDA is unavailable inside the allocated Slurm job." >&2
    exit 1
}

local_base=${LOCAL:-/tmp}
stage_root="$local_base/ncr-v7-$SLURM_JOB_ID"
dataset_root="$stage_root/data"
manifest="$stage_root/coco_unlabeled2017.jsonl"
export PIP_CACHE_DIR="$stage_root/pip-cache"
mkdir -p "$dataset_root" "$output_dir" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"

echo "Extracting packed COCO data on $(hostname) into node-local storage at $dataset_root"
unzip -q "$archive" -d "$dataset_root"

manifest_args=(
    --images "$dataset_root/unlabeled2017"
    --dataset-root "$dataset_root"
    --output "$manifest"
    --validation-fraction "${NCR_VALIDATION_FRACTION:-0.04}"
    --test-fraction "${NCR_TEST_FRACTION:-0.0}"
    --seed "${NCR_SEED:-2026}"
)
crop_size=$(python3 -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["training"]["crop_size"]))' "$config")
manifest_args+=(--min-size "$crop_size")
if [[ -n "${NCR_MAX_IMAGES:-}" ]]; then
    manifest_args+=(--max-images "$NCR_MAX_IMAGES")
fi
if [[ -f "$metadata" ]]; then
    manifest_args+=(--metadata "$metadata")
fi
python3 "$project_root/scripts/build_coco_manifest.py" "${manifest_args[@]}"

workers=${NCR_WORKERS:-${SLURM_CPUS_PER_TASK:-5}}
train_args=(
    --config "$config"
    --manifest "$manifest"
    --dataset-root "$dataset_root"
    --output-dir "$output_dir"
    --workers "$workers"
    --device cuda
    --skip-manifest-verification
)
if [[ -n "${NCR_BATCH_SIZE:-}" ]]; then
    train_args+=(--batch-size "$NCR_BATCH_SIZE")
fi
if [[ -n "${NCR_EPOCHS:-}" ]]; then
    train_args+=(--epochs "$NCR_EPOCHS")
fi
if [[ -n "$resume" ]]; then
    train_args+=(--resume "$resume")
fi

echo "Starting V7 job $SLURM_JOB_ID with config $config"
nvidia-smi
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" --cpu-bind=cores \
    python3 -u "$project_root/scripts/train_v7.py" "${train_args[@]}"
