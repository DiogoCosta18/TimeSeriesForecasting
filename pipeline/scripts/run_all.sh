#!/usr/bin/env bash
# Every stage of one configuration on one GPU machine (the pilot, protocol Section 8; or a
# single-machine run). GPU shards (neural, transformer) run in sequence; the CPU shards (ml;
# statistical in evaluation) run beside them. Each stage logs into RUN/logs/ and the run is
# uploaded to the bucket throughout (D20); the last upload is checked when the script ends.
#
#   scripts/run_all.sh CONFIG RUN_DIR DATA_DIR
#
# Run from the pipeline folder with the pinned environment's python on PATH (or PYTHON=...).
# STAT_WORKERS and ML_WORKERS (default 1) set evaluate --workers for the statistical and ML shards.
# DRY_RUN=1 prints the stage commands instead of running them. The script stops at the first
# failed stage (a failed gate included), after the sync has finished.
set -euo pipefail
CONFIG=${1:?usage: run_all.sh CONFIG RUN_DIR DATA_DIR}
RUN=${2:?usage: run_all.sh CONFIG RUN_DIR DATA_DIR}
DATA=${3:?usage: run_all.sh CONFIG RUN_DIR DATA_DIR}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GPU_SHARDS="neural-monthly neural-quarterly transformer-monthly transformer-quarterly"

stage() {  # stage LOG_NAME CLI_ARGS...
  if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "python -m src.cli ${*:2}"
  else
    "$HERE/run_stage.sh" "$RUN" "$@"
  fi
}

in_turn() {  # in_turn STAGE SHARD...: one stage over several shards, one after another
  local name=$1
  shift
  for shard in "$@"; do
    local extra=()
    if [ "$name" = evaluate ]; then
      case $shard in
        statistical-*) extra=(--workers "${STAT_WORKERS:-1}") ;;
        ml-*) extra=(--workers "${ML_WORKERS:-1}") ;;
      esac
    fi
    stage "$name-$shard" "$name" --config "$CONFIG" --run "$RUN" --data-dir "$DATA" --shard "$shard" "${extra[@]}"
  done
}

together() {  # together "STAGE SHARD..." ...: the groups in parallel; fails if any group fails
  local pids=() status=0 group
  for group in "$@"; do
    # shellcheck disable=SC2086
    in_turn $group &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  return "$status"
}

finish() {
  touch "$RUN/.sync_stop"
  wait "$SYNC" && echo "sync verified" || echo "SYNC FAILED: see $RUN.sync.log" >&2
}

if [ "${DRY_RUN:-0}" != 1 ]; then
  mkdir -p "$RUN"
  "$HERE/sync_run.sh" "$RUN" 300 > "$RUN.sync.log" 2>&1 &
  SYNC=$!
  trap finish EXIT
fi

stage prepare prepare --config "$CONFIG" --run "$RUN" --data-dir "$DATA" --jobs auto
together "tune $GPU_SHARDS" "tune ml-monthly ml-quarterly"
stage freeze freeze --config "$CONFIG" --run "$RUN"
together "evaluate $GPU_SHARDS" "evaluate ml-monthly ml-quarterly" "evaluate statistical-monthly statistical-quarterly"
stage merge merge --run "$RUN"
stage gates gates --run "$RUN"
stage analyse analyse --run "$RUN" --out "$RUN/analysis"
