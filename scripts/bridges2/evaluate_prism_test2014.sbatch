#!/usr/bin/env bash
# Submit from the repository with explicit checkpoint paths as arguments.
#SBATCH --account=cis260224p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --job-name=prism-test2014
#SBATCH --output=slurm_logs/prism-test2014-%j.out
#SBATCH --error=slurm_logs/prism-test2014-%j.err
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || $# -lt 1 ]]; then
    echo "Submit with one or more Prism checkpoint paths." >&2
    exit 2
fi
project_root=${NCR_CODE_ROOT:-$SLURM_SUBMIT_DIR}
storage=${NCR_PROJECT_ROOT:-/ocean/projects/cis260224p/shared/$USER}
cd "$project_root"
if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module purge
module load pytorch/26.05-2.11-py3
export PYTHONPATH="$project_root/.cluster/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
manifest="$storage/data/coco/manifests/prism_coco_test2014.jsonl"
test -f "$manifest"
stage_root=$(mktemp -d "${LOCAL:-/tmp}/prism-test2014-${SLURM_JOB_ID}.XXXXXX")
unzip -q "$storage/data/coco/test2014.zip" -d "$stage_root"
srun --ntasks=1 --cpu-bind=cores python3 scripts/evaluate_prism.py \
    --config research/configs/prism_coco_test2014.json \
    --weights "$@" --manifest "$manifest" --dataset-root "$stage_root" \
    --output-dir "$storage/evaluations/prism_coco_test2014/$SLURM_JOB_ID" \
    --split test --device cuda
