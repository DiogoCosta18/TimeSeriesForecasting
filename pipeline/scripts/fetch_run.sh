#!/usr/bin/env bash
# Download a run, or parts of it, from the bucket (protocol D20): to resume on a new machine
# or to give a machine the prepare bundle and the frozen configurations.
#
#   scripts/fetch_run.sh RUN_NAME DEST_DIR [PATH ...]
#
# PATHs are relative to the run (e.g. prepare configs_frozen.json tune/ml-monthly); without
# any, the whole run is fetched. Files are compared by checksum, and `rclone check` then
# confirms that everything fetched matches the bucket.
set -euo pipefail
RUN_NAME=${1:?usage: fetch_run.sh RUN_NAME DEST_DIR [PATH ...]}
DEST_DIR=${2:?usage: fetch_run.sh RUN_NAME DEST_DIR [PATH ...]}
shift 2
: "${RERUN_BUCKET:?set RERUN_BUCKET to the bucket name (never stored in the repository)}"
RCLONE=${RCLONE:-rclone}
REMOTE=${RERUN_REMOTE:-b2rerun}
SRC="$REMOTE:$RERUN_BUCKET/runs/$RUN_NAME"
FILTERS=()
for path in "$@"; do
  path=${path%/}
  FILTERS+=(--include "/$path" --include "/$path/**")
done
mkdir -p "$DEST_DIR"
"$RCLONE" copy "$SRC" "$DEST_DIR" --checksum --fast-list "${FILTERS[@]}" --log-level NOTICE
"$RCLONE" check "$SRC" "$DEST_DIR" --checksum --one-way --fast-list "${FILTERS[@]}" --log-level NOTICE
echo "$(date -u +%FT%TZ) fetched and verified: $SRC -> $DEST_DIR"
