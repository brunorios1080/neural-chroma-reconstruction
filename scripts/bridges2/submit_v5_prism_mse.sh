#!/usr/bin/env bash
# Submit a fresh or resumable matched V5 MSE run.
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
launcher="$project_root/scripts/bridges2/train_v5_prism_mse.sbatch"
log_dir="$project_root/slurm_logs"
mkdir -p "$log_dir"
command=(sbatch
    --export="ALL,NCR_CODE_ROOT=$project_root"
    --chdir="$project_root"
    --account="${NCR_ACCOUNT:-cis260224p}"
    --partition="${NCR_PARTITION:-GPU-shared}"
    --gres="gpu:${NCR_GPU:-l40s-48}:1"
    --cpus-per-task="${NCR_CPUS:-5}"
    --mem="${NCR_MEMORY:-62000M}"
    --time="${NCR_WALLTIME:-2-00:00:00}"
    --job-name=v5-prism-mse
    --output="$log_dir/v5-prism-mse-%j.out"
    --error="$log_dir/v5-prism-mse-%j.err"
    "$launcher")
if [[ "${1:-}" == "--dry-run" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
elif [[ $# -eq 0 ]]; then
    "${command[@]}"
else
    echo "Usage: bash scripts/bridges2/submit_v5_prism_mse.sh [--dry-run]" >&2
    exit 2
fi
