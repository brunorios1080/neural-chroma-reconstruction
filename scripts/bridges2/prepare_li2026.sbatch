#!/usr/bin/env bash
#SBATCH --account=cis260224p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=00:45:00
#SBATCH --export=ALL
#SBATCH --job-name=li2026-data
#SBATCH --output=slurm_logs/li2026-data-%j.out
#SBATCH --error=slurm_logs/li2026-data-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
python3 scripts/prepare_li2026_data.py /ocean/projects/cis260224p/shared/brios/data/li2026
