#!/usr/bin/env bash
# Arguments: frozen comparison directory. Array: Kodak, UHD 4K, UHD 6K, UHD 8K.
#SBATCH --account=cis260224p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=62000M
#SBATCH --time=04:00:00
#SBATCH --array=0-3%2
#SBATCH --export=ALL
#SBATCH --job-name=li2026-models
#SBATCH --output=slurm_logs/li2026-models-%A_%a.out
#SBATCH --error=slurm_logs/li2026-models-%A_%a.err
set -euo pipefail
case_root=$1
frozen_code=${3:-$case_root/code}
cd "${SLURM_SUBMIT_DIR:?}"
source /etc/profile.d/modules.sh
module purge
module load pytorch/26.05-2.11-py3
export PYTHONPATH="$frozen_code/vendor:$PWD/.cluster/python:$frozen_code${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
datasets=(kodak uhd240_4k uhd240_6k uhd240_8k)
dataset=${datasets[${SLURM_ARRAY_TASK_ID:?}]}
sampling=cosited_point
if [[ "$dataset" == kodak ]]; then sampling=cosited_point,center_box; fi
output="$case_root/groups/$dataset"
extra=()
if [[ "${2:-0}" != 0 ]]; then
    output="$case_root/probes/$dataset"
    extra=(--limit "$2")
fi
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=cores python3 "$frozen_code/scripts/evaluate_li2026.py" \
    --data /ocean/projects/cis260224p/shared/brios/data/li2026 \
    --output "$output" \
    --campaign /ocean/projects/cis260224p/shared/brios/evaluations/test2014_all_20260905/campaign.json \
    --datasets "$dataset" --sampling "$sampling" --matrix paper_rounded --device cuda "${extra[@]}"
if [[ "${2:-0}" == 0 ]]; then python3 "$frozen_code/scripts/collect_li2026.py" "$case_root"; fi
