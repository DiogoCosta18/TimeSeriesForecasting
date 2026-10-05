#!/usr/bin/env bash
# Run one stage with its log inside the run directory, so the log is synced with the outputs
# (protocol D20, Section 9.4).
#
#   scripts/run_stage.sh RUN_DIR LOG_NAME STAGE [ARGS ...]
#
# Runs `python -m src.cli STAGE ARGS` and appends its output to RUN_DIR/logs/LOG_NAME.log
# (also shown on the terminal); the exit code is the stage's.
set -euo pipefail
RUN_DIR=${1:?usage: run_stage.sh RUN_DIR LOG_NAME STAGE [ARGS ...]}
LOG_NAME=${2:?usage: run_stage.sh RUN_DIR LOG_NAME STAGE [ARGS ...]}
shift 2
PYTHON=${PYTHON:-python}
mkdir -p "$RUN_DIR/logs"
LOG="$RUN_DIR/logs/$LOG_NAME.log"
echo "=== $(date -u +%FT%TZ) $(hostname) python -m src.cli $*" >> "$LOG"
set +e
"$PYTHON" -m src.cli "$@" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
echo "=== $(date -u +%FT%TZ) exit $status" >> "$LOG"
exit "$status"
