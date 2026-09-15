#!/usr/bin/env bash
#SBATCH --account=cis260224p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=00:10:00
#SBATCH --exclude=v005
#SBATCH --export=ALL
#SBATCH --job-name=hybrid-pilot
#SBATCH --output=slurm_logs/hybrid-pilot-%j.out
#SBATCH --error=slurm_logs/hybrid-pilot-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source /etc/profile.d/modules.sh
module purge
module load pytorch/26.05-2.11-py3
export PYTHONPATH="/jet/home/brios/neural-chroma-reconstruction/.cluster/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HYBRID_REAL_CHECKPOINTS=1
python3 -m unittest tests.test_li2026_hybrid -v
python3 scripts/train_li2026_hybrid.py --config "${HYBRID_CONFIG:-research/configs/li2026_hybrid_pilot.json}" \
    --output "${HYBRID_OUTPUT:?Set HYBRID_OUTPUT to a new run directory}"
