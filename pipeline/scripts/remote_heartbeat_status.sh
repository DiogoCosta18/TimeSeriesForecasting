#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/workspace/outputs}"
LOG_FILE="$RUN_ROOT/VAST_HEARTBEAT.log"
mkdir -p "$RUN_ROOT"

while true; do
  {
    echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv || true
    else
      echo "nvidia-smi unavailable"
    fi
    df -h || true
    free -h || true
    STATUS_FILE="$(find "$RUN_ROOT" -name STATUS.json -print 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "$STATUS_FILE" && -f "$STATUS_FILE" ]]; then
      echo "--- latest STATUS.json: $STATUS_FILE ---"
      tail -n 120 "$STATUS_FILE" || true
    fi
    TRAIN_LOG="$(find "$RUN_ROOT" -path '*/logs/train.log' -print 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "$TRAIN_LOG" && -f "$TRAIN_LOG" ]]; then
      echo "--- latest train.log: $TRAIN_LOG ---"
      tail -n 80 "$TRAIN_LOG" || true
    fi
    echo "--- python processes ---"
    ps -eo pid,ppid,etimes,cmd | grep -E 'python|src.main|forecast' | grep -v grep || true
    echo
  } >> "$LOG_FILE" 2>&1
  sleep 60
done

