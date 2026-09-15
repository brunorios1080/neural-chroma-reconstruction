#!/usr/bin/env bash
# Args: frozen campaign directory, then 'preflight' or 'full'.
#SBATCH --account=cis260224p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62000M
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --job-name=coco2014-all
#SBATCH --output=slurm_logs/coco2014-all-%A_%a.out
#SBATCH --error=slurm_logs/coco2014-all-%A_%a.err
set -euo pipefail
campaign=$1
mode=${2:-full}
project_root=${NCR_CODE_ROOT:-$SLURM_SUBMIT_DIR}
storage=${NCR_PROJECT_ROOT:-/ocean/projects/cis260224p/shared/$USER}
if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module purge
module load pytorch/26.05-2.11-py3
export PYTHONPATH="$project_root/.cluster/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$project_root"
stage=$(mktemp -d "${LOCAL:-/tmp}/coco2014-all-${SLURM_JOB_ID}.XXXXXX")
echo "Staging COCO test2014 on $(hostname)"
unzip -q "$storage/data/coco/test2014.zip" -d "$stage"
if [[ "$mode" == preflight ]]; then
    group=-1
    output="$campaign/preflight"
    extra=(--limit 16)
else
    group=${SLURM_ARRAY_TASK_ID:?Full testing requires an array task index}
    output="$campaign/groups/$group"
    extra=()
fi
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --cpu-bind=cores python3 "$campaign/code/scripts/compare_available_models.py" \
    --campaign "$campaign/campaign.json" --dataset-root "$stage" --output "$output" \
    --group "$group" --batch-size 16 --workers 6 --device cuda "${extra[@]}"
if [[ "$mode" != preflight ]]; then
    python3 "$campaign/code/scripts/collect_model_comparison.py" "$campaign"
fi
