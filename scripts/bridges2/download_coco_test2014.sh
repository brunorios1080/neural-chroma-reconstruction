#!/usr/bin/env bash
# Run in an allocated compute job; retain archives on shared storage.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this download inside a Slurm allocation." >&2
    exit 2
fi
storage=${NCR_PROJECT_ROOT:-/ocean/projects/cis260224p/shared/$USER}
destination="$storage/data/coco"
mkdir -p "$destination/annotations"
fetch() {
    local url=$1 target=$2 expected=$3
    if [[ -f "$target" ]]; then
        test "$(stat -c %s "$target")" = "$expected"
        echo "Already downloaded: $target"
        return
    fi
    echo "Downloading $url"
    curl --fail --location --silent --show-error --retry 5 --connect-timeout 30 \
        --max-time 1200 --continue-at - --output "$target.part" "$url"
    test "$(stat -c %s "$target.part")" = "$expected"
    mv "$target.part" "$target"
    echo "Downloaded $target ($expected bytes)"
}
# TLS endpoint for the official images.cocodataset.org S3 bucket.
fetch https://s3.amazonaws.com/images.cocodataset.org/zips/test2014.zip \
    "$destination/test2014.zip" 6660437059
fetch https://s3.amazonaws.com/images.cocodataset.org/annotations/image_info_test2014.zip \
    "$destination/annotations/image_info_test2014.zip" 763464
