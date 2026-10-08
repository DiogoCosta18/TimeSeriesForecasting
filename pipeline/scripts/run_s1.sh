#!/usr/bin/env bash
# Sensitivity analysis S1 (protocol change log v2.2) on one rented machine: tuning (the cpu and
# gpu groups side by side), freeze, then evaluation (both groups side by side). The analysis
# runs where the run's merged rows are, with `python -m src.stages.sensitivity analyse`.
# The run directory is uploaded throughout and checked at the end (D20); ON_DONE (e.g. stopping
# the machine) runs only after success and a checked last upload.
#
#   scripts/run_s1.sh CONFIG RUN_DIR DATA_DIR
#
# RUN_DIR holds the run's prepare bundle (scripts/fetch_run.sh RUN prepare). CPU_WORKERS
# (default 8) processes evaluate the cpu group. PYTHON, RERUN_BUCKET, RERUN_REMOTE and RCLONE as
# for the other scripts. Logs: RUN_DIR/logs/s1-*.log.
set -euo pipefail
CONFIG=${1:?usage: run_s1.sh CONFIG RUN_DIR DATA_DIR}
RUN=${2:?usage: run_s1.sh CONFIG RUN_DIR DATA_DIR}
DATA=${3:?usage: run_s1.sh CONFIG RUN_DIR DATA_DIR}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
mkdir -p "$RUN/logs"

s1() {  # s1 LOG_NAME STAGE ARGS...: one S1 stage, logged in the run
  local log="$RUN/logs/$1.log" status=0
  shift
  echo "=== $(date -u +%FT%TZ) $(hostname) python -m src.stages.sensitivity $*" >> "$log"
  "$PYTHON" -m src.stages.sensitivity "$@" >> "$log" 2>&1 || status=$?
  echo "=== $(date -u +%FT%TZ) exit $status" >> "$log"
  return "$status"
}

both() {  # both STAGE EXTRA_CPU_ARGS: the stage for the cpu and gpu groups side by side
  local stage=$1 status=0 pids=()
  s1 "s1-$stage-cpu" "$stage" --config "$CONFIG" --run "$RUN" --data-dir "$DATA" --group cpu $2 &
  pids+=($!)
  s1 "s1-$stage-gpu" "$stage" --config "$CONFIG" --run "$RUN" --data-dir "$DATA" --group gpu &
  pids+=($!)
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  return "$status"
}

finish() {
  local status=$?
  touch "$RUN/.sync_stop"
  if wait "$SYNC"; then echo "last upload checked"; else echo "SYNC FAILED: see $RUN.sync.log" >&2; status=1; fi
  if [ "$status" -eq 0 ] && [ -n "${ON_DONE:-}" ]; then
    echo "S1 done; running ON_DONE"
    bash -c "$ON_DONE" || echo "ON_DONE failed" >&2
  fi
  exit "$status"
}

rm -f "$RUN/.sync_stop"
"$HERE/sync_run.sh" "$RUN" 300 > "$RUN.sync.log" 2>&1 &
SYNC=$!
trap finish EXIT

both tune ""
s1 s1-freeze freeze --config "$CONFIG" --run "$RUN"
both evaluate "--workers ${CPU_WORKERS:-8}"
echo "S1 tuning and evaluation finished"
