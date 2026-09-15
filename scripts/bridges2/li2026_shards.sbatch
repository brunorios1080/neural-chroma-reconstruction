#!/usr/bin/env bash
#SBATCH --account=cis260224p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=62000M
#SBATCH --time=01:00:00
#SBATCH --array=0-15%8
#SBATCH --export=ALL
#SBATCH --job-name=li2026-uhd
#SBATCH --output=slurm_logs/li2026-uhd-%A_%a.out
#SBATCH --error=slurm_logs/li2026-uhd-%A_%a.err
set -euo pipefail
case_root=$1
frozen_code="$case_root/code_sharded"
cd "${SLURM_SUBMIT_DIR:?}"
source /etc/profile.d/modules.sh
module purge
module load pytorch/26.05-2.11-py3
export PYTHONPATH="$frozen_code/vendor:$PWD/.cluster/python:$frozen_code${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=cores python3 "$frozen_code/scripts/evaluate_li2026.py" \
    --data /ocean/projects/cis260224p/shared/brios/data/li2026 \
    --output "$case_root/shards/${SLURM_ARRAY_TASK_ID:?}" \
    --campaign /ocean/projects/cis260224p/shared/brios/evaluations/test2014_all_20260905/campaign.json \
    --datasets uhd240 --sampling cosited_point --matrix paper_rounded --device cuda \
    --shard-count 16 --shard-index "$SLURM_ARRAY_TASK_ID"
python3 "$frozen_code/scripts/collect_li2026.py" "$case_root"
