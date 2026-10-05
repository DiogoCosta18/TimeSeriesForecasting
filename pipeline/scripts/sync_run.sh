#!/usr/bin/env bash
# Upload a run directory to the bucket while stages run (protocol D20, Section 9.4).
#
#   scripts/sync_run.sh RUN_DIR [INTERVAL_SECONDS]
#
# Every INTERVAL seconds (default 300) RUN_DIR is copied, with checksums, to
# $RERUN_REMOTE:$RERUN_BUCKET/runs/<name of RUN_DIR>/ (remote default: b2rerun). It stops when
# RUN_DIR/.sync_stop exists: a last copy, then `rclone check` that every local file is in the
# bucket with the same checksum; the exit code is non-zero if that fails.
# `rclone copy` never deletes anything in the bucket, so no machine can remove another
# machine's outputs. Temporary files of atomic writes (.*.tmp) are not uploaded. A failed
# intermediate copy (network) is reported and retried at the next interval. Each call lists the
# bucket once, recursively (--fast-list), to keep B2 transactions few.
set -euo pipefail
RUN_DIR=${1:?usage: sync_run.sh RUN_DIR [INTERVAL_SECONDS]}
INTERVAL=${2:-300}
: "${RERUN_BUCKET:?set RERUN_BUCKET to the bucket name (never stored in the repository)}"
RCLONE=${RCLONE:-rclone}
REMOTE=${RERUN_REMOTE:-b2rerun}
RUN_DIR=$(realpath "$RUN_DIR")
DEST="$REMOTE:$RERUN_BUCKET/runs/$(basename "$RUN_DIR")"
FILTERS=(--exclude ".*.tmp" --exclude "/.sync_stop")

copy() { "$RCLONE" copy "$RUN_DIR" "$DEST" --checksum --fast-list "${FILTERS[@]}" --log-level NOTICE; }

while [ ! -e "$RUN_DIR/.sync_stop" ]; do
  copy || echo "$(date -u +%FT%TZ) sync to $DEST failed; retrying in ${INTERVAL}s" >&2
  waited=0
  while [ "$waited" -lt "$INTERVAL" ] && [ ! -e "$RUN_DIR/.sync_stop" ]; do
    sleep 1
    waited=$((waited + 1))
  done
done
copy
"$RCLONE" check "$RUN_DIR" "$DEST" --checksum --one-way --fast-list "${FILTERS[@]}" --log-level NOTICE
echo "$(date -u +%FT%TZ) final sync verified: $DEST"
