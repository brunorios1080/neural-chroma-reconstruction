#!/usr/bin/env bash
# Submit one independent Bridges-2 GPU job per V7 config.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 CONFIG.json [CONFIG.json ...]" >&2
    exit 2
fi

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
launcher="$project_root/scripts/bridges2/train_v7.sbatch"
log_dir="$project_root/slurm_logs"
mkdir -p "$log_dir"

account=${NCR_ACCOUNT:-cis260224p}
partition=${NCR_PARTITION:-GPU-shared}
gpu=${NCR_GPU:-l40s-48}
cpus=${NCR_CPUS:-8}
memory=${NCR_MEMORY:-62000M}
walltime=${NCR_WALLTIME:-1-00:00:00}

for config in "$@"; do
    if [[ "$config" != /* ]]; then
        config="$project_root/$config"
    fi
    if [[ ! -f "$config" ]]; then
        echo "Missing config: $config" >&2
        exit 1
    fi
    stem=$(basename "$config" .json)
    stem=$(printf '%s' "$stem" | tr -c '[:alnum:]_-' '-')
    sbatch \
        --export="ALL,NCR_CODE_ROOT=$project_root" \
        --account="$account" \
        --partition="$partition" \
        --gres="gpu:$gpu:1" \
        --cpus-per-task="$cpus" \
        --mem="$memory" \
        --time="$walltime" \
        --job-name="ncr-$stem" \
        --output="$log_dir/%x-%j.out" \
        --error="$log_dir/%x-%j.err" \
        "$launcher" "$config"
done
