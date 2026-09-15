#!/usr/bin/env bash
# Submit independent Prism experiments as a bounded-concurrency job array.
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
config="$project_root/research/configs/prism_coco.json"
dry_run=false
models=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) config=$2; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --help|-h)
            echo "Usage: bash scripts/bridges2/submit_prism.sh [--dry-run] [--config FILE] [prism_NAME ...]"
            exit 0 ;;
        --*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) models+=("$1"); shift ;;
    esac
done
if [[ "$config" != /* ]]; then config="$PWD/$config"; fi
# Bridges-2's system python3 may be too old for this repository.
python=${NCR_PYTHON:-/opt/packages/AI/pytorch_26.05-py3/bin/python3}
all_names=$("$python" "$project_root/scripts/train_prism.py" --config "$config" --list)
mapfile -t available <<< "$all_names"
if [[ ${#models[@]} -eq 0 ]]; then models=("${available[@]}"); fi
declare -A chosen=()
for name in "${models[@]}"; do
    found=false
    for allowed in "${available[@]}"; do
        if [[ "$name" == "$allowed" ]]; then found=true; fi
    done
    if [[ "$found" != true || -n "${chosen[$name]:-}" ]]; then
        echo "Unknown or duplicate Prism experiment: $name" >&2
        exit 2
    fi
    chosen[$name]=1
done
concurrency=${NCR_MAX_CONCURRENT:-2}
if [[ ! "$concurrency" =~ ^[1-9][0-9]*$ ]]; then echo "NCR_MAX_CONCURRENT must be positive" >&2; exit 2; fi
log_dir="$project_root/slurm_logs"
dependency=()
if [[ -n "${NCR_DEPENDENCY:-}" ]]; then
    dependency=(--dependency="$NCR_DEPENDENCY" --kill-on-invalid-dep=yes)
fi
command=(sbatch
    --export=ALL
    --chdir="$project_root"
    --account="${NCR_ACCOUNT:-cis260224p}"
    --partition="${NCR_PARTITION:-GPU-shared}"
    --gres="gpu:${NCR_GPU:-l40s-48}:1"
    --cpus-per-task="${NCR_CPUS:-8}"
    --mem="${NCR_MEMORY:-62000M}"
    --time="${NCR_WALLTIME:-08:00:00}"
    --array="0-$((${#models[@]}-1))%$concurrency"
    --job-name=prism
    --output="$log_dir/prism-%A_%a.out"
    --error="$log_dir/prism-%A_%a.err"
    "${dependency[@]}"
    "$project_root/scripts/bridges2/train_prism.sbatch" "$config" "${models[@]}")
for index in "${!models[@]}"; do printf 'Array task %s: %s\n' "$index" "${models[$index]}"; done
echo "Each task requests one GPU; walltime cap ${NCR_WALLTIME:-08:00:00}; concurrency $concurrency."
if [[ "$dry_run" == true ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    mkdir -p "$log_dir"
    "${command[@]}"
fi
