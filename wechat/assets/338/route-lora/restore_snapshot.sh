#!/usr/bin/env bash
set -euo pipefail
study_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
target=${1:?Usage: restore_snapshot.sh TARGET_DIRECTORY}
mkdir -p -- "$target"
if [ -n "$(ls -A -- "$target")" ]; then
    echo 'Target directory must be empty.' >&2
    exit 1
fi
python3 "$study_dir/verify_snapshot.py" "$study_dir/snapshot"
cat "$study_dir"/snapshot/*.zst.part-* | zstd -dc | tar -xf - -C "$target"
