#!/usr/bin/env bash
#SBATCH --account=cis260224p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=62000M
#SBATCH --time=00:05:00
#SBATCH --export=ALL
#SBATCH --job-name=li2026-metrics
#SBATCH --output=slurm_logs/li2026-metrics-%j.out
#SBATCH --error=slurm_logs/li2026-metrics-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source /etc/profile.d/modules.sh
module purge
module load pytorch/26.05-2.11-py3
export PYTHONPATH="$PWD/.cluster/li2026:$PWD/.cluster/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python3 scripts/preflight_li2026_metrics.py /ocean/projects/cis260224p/shared/brios/evaluations/li2026_20260905/metrics_preflight.json
