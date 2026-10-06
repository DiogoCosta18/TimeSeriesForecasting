#!/usr/bin/env bash
# One machine's part of a run on several machines (protocol Section 9.1, D20). The machines
# coordinate only through the bucket, so the run needs no operator between stages: every step
# publishes its outputs (copied, then checked file by file against the bucket) and then writes a
# marker run/coordination/STEP.done.json; a machine that needs another's outputs waits for its
# marker, then fetches and checks them.
#
#   scripts/run_machine.sh CONFIG RUN_DIR DATA_DIR
#
# Environment (shard lists are space-separated SHARD[:WORKERS] entries, run one after another):
#   MACHINE      name used in the logs and markers (default: hostname)
#   LEAD=1       this machine runs prepare, freeze, and merge + gates + analyse at the end
#   TUNE         tuning shards of this machine            e.g. "ml-monthly:20 ml-quarterly:20"
#   EVALUATE     evaluation shards; groups separated by "|" run side by side
#                                                         e.g. "ml-monthly:8 ml-quarterly:8 | statistical-monthly:20"
#   POLL         seconds between looks at the bucket (default 120); WAIT_HOURS gives up (default 48)
#   ON_DONE      command run after everything succeeded and the last upload was checked (e.g. stop
#                the rented machine); never run after a failure, so a failed machine stays up
#   RERUN_BUCKET, RERUN_REMOTE, RCLONE, PYTHON as for sync_run.sh, fetch_run.sh and run_stage.sh
#
# Every machine uses the same RUN_DIR name (it names the run in the bucket). A failed step writes
# STEP.failed.json and stops this machine; the other machines stop waiting when they see a failed
# step without a later done marker. Steps whose done marker exists are skipped, so the script can
# be started again after an interruption (the stages themselves resume task by task).
set -euo pipefail
CONFIG=${1:?usage: run_machine.sh CONFIG RUN_DIR DATA_DIR}
RUN=${2:?usage: run_machine.sh CONFIG RUN_DIR DATA_DIR}
DATA=${3:?usage: run_machine.sh CONFIG RUN_DIR DATA_DIR}
: "${RERUN_BUCKET:?set RERUN_BUCKET to the bucket name (never stored in the repository)}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RCLONE=${RCLONE:-rclone}
REMOTE=${RERUN_REMOTE:-b2rerun}
MACHINE=${MACHINE:-$(hostname)}
POLL=${POLL:-120}
WAIT_HOURS=${WAIT_HOURS:-48}
mkdir -p "$RUN/logs"
RUN=$(realpath "$RUN")
NAME=$(basename "$RUN")
DEST="$REMOTE:$RERUN_BUCKET/runs/$NAME"
LOG="$RUN/logs/machine-$MACHINE.log"
# Every shard of the protocol (src.stages.tasks; test_scripts checks these lists against it).
TUNE_SHARDS_ALL="ml-monthly ml-quarterly neural-monthly neural-quarterly transformer-monthly transformer-quarterly"
EVALUATE_SHARDS_ALL="statistical-monthly statistical-quarterly $TUNE_SHARDS_ALL"

log() { echo "$(date -u +%FT%TZ) [$MACHINE] $*" | tee -a "$LOG" >&2; }

retry() {  # retry CMD...: up to 5 attempts, 30 s apart (network)
  local n
  for n in 1 2 3 4 5; do
    "$@" && return 0
    log "attempt $n failed: $*"
    [ "$n" -lt 5 ] && sleep "${RETRY_SLEEP:-30}"
  done
  return 1
}

markers() {  # the marker names; coordination/ exists from this machine's first marker on
  local out
  if ! out=$(retry "$RCLONE" lsf "$DEST/coordination" 2>> "$LOG"); then
    log "cannot list the bucket; stopping"
    exit 1
  fi
  printf '%s\n' "$out"
}

is_done() {
  local have
  have=$(markers) || exit 1
  grep -qx "$1.done.json" <<<"$have"
}

mark() {  # mark STEP STATUS: coordination/STEP.STATUS.json, here and in the bucket
  local file="$RUN/coordination/$1.$2.json"
  mkdir -p "$RUN/coordination"
  printf '{"step": "%s", "status": "%s", "machine": "%s", "host": "%s", "utc": "%s"}\n' \
    "$1" "$2" "$MACHINE" "$(hostname)" "$(date -u +%FT%TZ)" > "$file"
  retry "$RCLONE" copyto "$file" "$DEST/coordination/$1.$2.json" --log-level NOTICE
  log "marked $1 $2"
}

publish() {  # publish PATH...: copy run paths to the bucket, then check every file against it
  # Ordered rules, the first match wins: never temporaries of atomic writes, then the paths,
  # nothing else (rclone puts all --include rules before any --exclude, so they cannot be mixed).
  local filters=(--filter "- .*.tmp") p
  for p in "$@"; do filters+=(--filter "+ /$p" --filter "+ /$p/**"); done
  filters+=(--filter "- **")
  retry "$RCLONE" copy "$RUN" "$DEST" --checksum "${filters[@]}" --log-level NOTICE
  retry "$RCLONE" check "$RUN" "$DEST" --checksum --one-way "${filters[@]}" --log-level NOTICE
}

fetch() { retry "$HERE/fetch_run.sh" "$NAME" "$RUN" "$@" >> "$LOG" 2>&1; }

wait_for() {  # wait_for STEP...: until every STEP is done; fails if a step failed (and was not redone)
  local start missing failed have step
  start=$(date +%s)
  log "waiting for: $*"
  while :; do
    have=$(markers) || exit 1
    failed=()
    for step in $(sed -n 's/\.failed\.json$//p' <<<"$have"); do
      grep -qx "$step.done.json" <<<"$have" || failed+=("$step")
    done
    if [ ${#failed[@]} -gt 0 ]; then
      log "stopping: failed elsewhere: ${failed[*]}"
      return 1
    fi
    missing=()
    for step in "$@"; do grep -qx "$step.done.json" <<<"$have" || missing+=("$step"); done
    [ ${#missing[@]} -eq 0 ] && { log "ready: $*"; return 0; }
    if [ $(( $(date +%s) - start )) -ge $(( WAIT_HOURS * 3600 )) ]; then
      log "gave up after ${WAIT_HOURS} h waiting for: ${missing[*]}"
      return 1
    fi
    sleep "$POLL"
  done
}

step() {  # step STEP LOG_NAME CLI_ARGS...: one stage; marks STEP failed if it fails
  local name=$1 log_name=$2
  shift 2
  log "start $name"
  if "$HERE/run_stage.sh" "$RUN" "$log_name" "$@" > /dev/null; then
    log "end $name"
  else
    log "FAILED $name (see logs/$log_name.log)"
    mark "$name" failed
    publish "logs/$log_name.log" || true
    return 1
  fi
}

shards() {  # shards STAGE "SHARD[:WORKERS] ...": one after another, each published and marked
  local stage=$1 entry shard workers
  for entry in $2; do
    shard=${entry%%:*}
    workers=1
    [ "$entry" != "$shard" ] && workers=${entry#*:}
    is_done "$stage-$shard" && { log "skip $stage-$shard (done)"; continue; }
    step "$stage-$shard" "$stage-$shard" "$stage" --config "$CONFIG" --run "$RUN" --data-dir "$DATA" \
      --shard "$shard" --workers "$workers" || return 1
    publish "$stage/$shard" "logs/$stage-$shard.log" || { mark "$stage-$shard" failed; return 1; }
    mark "$stage-$shard" done
  done
}

finish() {
  local status=$?
  touch "$RUN/.sync_stop"
  if wait "$SYNC"; then log "last upload checked"; else log "SYNC FAILED: see $RUN.sync.log"; status=1; fi
  if [ "$status" -eq 0 ] && [ -n "${ON_DONE:-}" ]; then
    log "all done; running ON_DONE"
    bash -c "$ON_DONE" >> "$LOG" 2>&1 || log "ON_DONE failed"
  fi
  exit "$status"
}

rm -f "$RUN/.sync_stop"
"$HERE/sync_run.sh" "$RUN" 300 > "$RUN.sync.log" 2>&1 &
SYNC=$!
trap finish EXIT
mark "machine-$MACHINE" started
log "start: lead=${LEAD:-0} tune='${TUNE:-}' evaluate='${EVALUATE:-}' commit $(git -C "$HERE" rev-parse --short HEAD)"

# R1 prepare: the lead prepares, the others fetch the bundle
if [ "${LEAD:-0}" = 1 ]; then
  if ! is_done prepare; then
    step prepare prepare prepare --config "$CONFIG" --run "$RUN" --data-dir "$DATA" --jobs auto
    publish prepare logs/prepare.log || { mark prepare failed; exit 1; }
    mark prepare done
  fi
else
  wait_for prepare
  fetch prepare
fi

# R2 tune this machine's shards
shards tune "${TUNE:-}"

# R2b freeze: the lead collects every study; the others fetch the frozen configurations
tune_steps=()
for shard in $TUNE_SHARDS_ALL; do tune_steps+=("tune-$shard"); done
if [ "${LEAD:-0}" = 1 ]; then
  if ! is_done freeze; then
    wait_for "${tune_steps[@]}"
    fetch tune
    step freeze freeze freeze --config "$CONFIG" --run "$RUN"
    publish configs_frozen.json logs/freeze.log || { mark freeze failed; exit 1; }
    mark freeze done
  fi
elif [ -n "${EVALUATE:-}" ]; then
  wait_for freeze
  fetch configs_frozen.json
fi

# R3 evaluate this machine's shards; "|" separates groups that run side by side
if [ -n "${EVALUATE:-}" ]; then
  pids=()
  IFS='|' read -r -a groups <<<"$EVALUATE"
  for group in "${groups[@]}"; do
    shards evaluate "$group" &
    pids+=($!)
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  [ "$status" -eq 0 ] || exit 1
fi

# R4 merge, gates and analysis on the lead, once every shard is evaluated
if [ "${LEAD:-0}" = 1 ] && ! is_done analyse; then
  eval_steps=()
  for shard in $EVALUATE_SHARDS_ALL; do eval_steps+=("evaluate-$shard"); done
  wait_for "${eval_steps[@]}"
  fetch evaluate
  step merge merge merge --run "$RUN"
  step gates gates gates --run "$RUN"
  step analyse analyse analyse --run "$RUN" --out "$RUN/analysis"
  publish merged analysis logs/merge.log logs/gates.log logs/analyse.log || { mark analyse failed; exit 1; }
  mark analyse done
fi
log "finished"
