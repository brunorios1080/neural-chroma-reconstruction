#!/usr/bin/env bash
#SBATCH --account=cis260224p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --time=1-00:00:00
#SBATCH --job-name=ncr-v5.1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch; do not train on a login node." >&2
    exit 2
fi

code_root=${NCR_CODE_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$code_root" || ! -f "$code_root/scripts/train.py" ]]; then
    echo "Cannot locate the neural-chroma-reconstruction code checkout." >&2
    echo "Submit with scripts/bridges2/submit_v5_1.sh or set NCR_CODE_ROOT." >&2
    exit 1
fi

storage_root=${NCR_PROJECT_ROOT:-/ocean/projects/cis260224p/shared/$USER}
archive=${NCR_DATASET_ARCHIVE:-$storage_root/data/coco/unlabeled2017.zip}
initial_checkpoint=${NCR_INITIAL_CHECKPOINT:-$code_root/models/version5/epoch_010.pth}
output_dir=${NCR_OUTPUT_DIR:-$storage_root/checkpoints/v5.1}
samples_dir=${NCR_SAMPLES_DIR:-$storage_root/samples/v5.1}
if [[ -n "${NCR_MAX_IMAGES:-}" && -z "${NCR_OUTPUT_DIR:-}" ]]; then
    output_dir="${output_dir}_smoke${NCR_MAX_IMAGES}"
    samples_dir="${samples_dir}_smoke${NCR_MAX_IMAGES}"
fi

for required in "$archive" "$initial_checkpoint"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required V5.1 input: $required" >&2
        exit 1
    fi
done

if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
fi
module purge
module load pytorch/26.05-2.11-py3

export PYTHONPATH="$code_root/.cluster/python${PYTHONPATH:+:$PYTHONPATH}"
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
stage_root="$local_base/ncr-v5.1-$SLURM_JOB_ID"
dataset_root="$stage_root/data"
export PIP_CACHE_DIR="$stage_root/pip-cache"
mkdir -p "$dataset_root" "$output_dir" "$samples_dir" "$PIP_CACHE_DIR"

echo "Extracting packed COCO data on $(hostname) into $dataset_root"
unzip -q "$archive" -d "$dataset_root"

resume=${NCR_RESUME:-}
if [[ -z "$resume" ]]; then
    if [[ -f "$output_dir/last.pth" ]]; then
        resume="$output_dir/last.pth"
    else
        resume="$initial_checkpoint"
    fi
fi

train_args=(
    --src "$dataset_root/unlabeled2017"
    --epochs "${NCR_EPOCHS:-100}"
    --batch-size "${NCR_BATCH_SIZE:-64}"
    --crop 256
    --workers "${NCR_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
    --val-fraction 0.03
    --seed 1337
    --learning-rate 2e-4
    --device cuda
    --resume "$resume"
    --output-dir "$output_dir"
    --samples-dir "$samples_dir"
    --run-name v5.1
    --no-amp
)
if [[ -n "${NCR_MAX_IMAGES:-}" ]]; then
    train_args+=(--max-images "$NCR_MAX_IMAGES")
fi

echo "Starting V5.1 job $SLURM_JOB_ID from $resume"
nvidia-smi
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" --cpu-bind=cores \
    python3 -u "$code_root/scripts/train.py" --model v5 "${train_args[@]}"
