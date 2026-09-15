#!/usr/bin/env bash
#SBATCH --account=cis260224p
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=00:20:00
#SBATCH --export=ALL
#SBATCH --job-name=li2026-check
#SBATCH --output=slurm_logs/li2026-check-%j.out
#SBATCH --error=slurm_logs/li2026-check-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source /etc/profile.d/modules.sh
module purge
module load pytorch/26.05-2.11-py3
export PYTHONPATH="$PWD/.cluster/li2026:$PWD/.cluster/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
uv pip install --no-deps --target "$PWD/.cluster/li2026" --cache-dir "${LOCAL:-/tmp}/li2026-uv" piq==0.8.0
python3 -m unittest tests.test_li2026 -v
campaign=/ocean/projects/cis260224p/shared/brios/evaluations/test2014_all_20260905/campaign.json
output=/ocean/projects/cis260224p/shared/brios/evaluations/li2026_20260905
python3 scripts/preflight_li2026.py "$campaign" "$output/preflight.json"
python3 scripts/evaluate_li2026.py --data /ocean/projects/cis260224p/shared/brios/data/li2026 \
    --output "$output/calibration" --mode calibrate
