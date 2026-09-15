#!/usr/bin/env bash
# Submit the V5.1 continuation on Bridges-2.

set -euo pipefail

code_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
launcher="$code_root/scripts/bridges2/train_v5_1.sbatch"
log_dir="$code_root/slurm_logs"
mkdir -p "$log_dir"

account=${NCR_ACCOUNT:-cis260224p}
partition=${NCR_PARTITION:-GPU-shared}
gpu=${NCR_GPU:-l40s-48}
cpus=${NCR_CPUS:-8}
memory=${NCR_MEMORY:-62000M}
walltime=${NCR_WALLTIME:-1-00:00:00}

sbatch \
    --export="ALL,NCR_CODE_ROOT=$code_root" \
    --account="$account" \
    --partition="$partition" \
    --gres="gpu:$gpu:1" \
    --cpus-per-task="$cpus" \
    --mem="$memory" \
    --time="$walltime" \
    --job-name=ncr-v5.1 \
    --output="$log_dir/%x-%j.out" \
    --error="$log_dir/%x-%j.err" \
    "$launcher"
