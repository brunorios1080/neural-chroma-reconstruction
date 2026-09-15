#!/usr/bin/env bash
# Invoked by submit_prism.sh; each task stages data on its allocated GPU node.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || $# -lt 2 ]]; then
    echo "Use scripts/bridges2/submit_prism.sh to allocate a GPU and select models." >&2
    exit 2
fi
project_root=${NCR_CODE_ROOT:-$SLURM_SUBMIT_DIR}
config=$1
shift
models=("$@")
index=${SLURM_ARRAY_TASK_ID:-0}
if [[ "$index" -ge "${#models[@]}" ]]; then echo "Invalid array index" >&2; exit 2; fi
model=${models[$index]}
storage_root=${NCR_PROJECT_ROOT:-/ocean/projects/cis260224p/shared/$USER}
if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module purge
module load pytorch/26.05-2.11-py3
export PYTHONPATH="$project_root/.cluster/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python3 -c 'import torch; assert torch.cuda.is_available(), "Allocated task has no usable CUDA device"'
cd "$project_root"
settings=$(python3 - "$config" <<'PY'
import sys
from pathlib import Path
from chroma.prism_config import load_suite
c = load_suite(sys.argv[1])
for value in (c['data_kind'], c['training']['crop_size'], c['training']['seed'], c['manifest'], c['dataset_root']):
    print(value)
print(c.get('max_images') or '')
PY
)
mapfile -t fields <<< "$settings"
kind=${fields[0]}
crop=${fields[1]}
seed=${fields[2]}
manifest=${NCR_MANIFEST:-${fields[3]}}
dataset_root=${NCR_DATASET_ROOT:-${fields[4]}}
max_images=${NCR_MAX_IMAGES:-${fields[5]:-}}
if [[ -n "$max_images" && ( ! "$max_images" =~ ^[1-9][0-9]*$ || "$max_images" -lt 50 ) ]]; then
    echo "Prism max_images must be an integer >=50" >&2
    exit 2
fi
stem=$(basename "$config" .json)
output_root=${NCR_OUTPUT_ROOT:-$storage_root/checkpoints/prism/$stem}
if [[ -n "$max_images" ]]; then output_root="${output_root}_smoke${max_images}"; fi
extra=()
if [[ "$kind" == "synthetic_coco" && -z "${NCR_MANIFEST:-}" ]]; then
    archive=${NCR_DATASET_ARCHIVE:-$storage_root/data/coco/unlabeled2017.zip}
    metadata=${NCR_COCO_METADATA:-$storage_root/data/coco/annotations/image_info_unlabeled2017.json}
    if [[ ! -f "$archive" ]]; then echo "Missing archive: $archive" >&2; exit 2; fi
    stage_root=$(mktemp -d "${LOCAL:-/tmp}/prism-${SLURM_JOB_ID}-${index}.XXXXXX")
    dataset_root="$stage_root/data"
    manifest="$stage_root/prism_coco.jsonl"
    mkdir -p "$dataset_root"
    unzip -q "$archive" -d "$dataset_root"
    prepare=(python3 scripts/prepare_prism_coco.py --images "$dataset_root/unlabeled2017" --dataset-root "$dataset_root" --output "$manifest" --min-size "$crop" --seed "$seed")
    if [[ -f "$metadata" ]]; then prepare+=(--metadata "$metadata"); fi
    if [[ -n "$max_images" ]]; then prepare+=(--max-images "$max_images"); fi
    "${prepare[@]}"
    extra+=(--skip-hash-verification)
fi
if [[ -n "$max_images" ]]; then
    python3 - "$manifest" "$max_images" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

records = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
limit = int(sys.argv[2])
if not 50 <= len(records) <= limit:
    raise SystemExit(f"Refusing smoke training: manifest has {len(records)} images; expected 50..{limit}")
print(json.dumps({'smoke_max_images': limit, 'manifest_images': len(records),
                  'splits': dict(Counter(row['split'] for row in records))}), flush=True)
PY
fi
if [[ -n "${NCR_EPOCHS:-}" ]]; then extra+=(--epochs "$NCR_EPOCHS"); fi
if [[ -n "${NCR_STOP_AFTER_EPOCH:-}" ]]; then extra+=(--stop-after-epoch "$NCR_STOP_AFTER_EPOCH"); fi
if [[ -n "${NCR_BATCH_SIZE:-}" ]]; then extra+=(--batch-size "$NCR_BATCH_SIZE"); fi
echo "Training $model for job $SLURM_JOB_ID on $(hostname); output root $output_root"
nvidia-smi
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --cpu-bind=cores \
    python3 -u scripts/train_prism.py --config "$config" --model "$model" \
    --manifest "$manifest" --dataset-root "$dataset_root" --output-root "$output_root" \
    --workers "${NCR_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}" --device cuda --resume auto "${extra[@]}"
