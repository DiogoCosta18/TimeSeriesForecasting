#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/workspace/outputs}"
while true; do
  date
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv || true
  else
    echo "nvidia-smi unavailable"
  fi
  df -h || true
  STATUS="$(find "$RUN_ROOT" -name STATUS.json -print 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${STATUS}" && -f "${STATUS}" ]]; then
    tail -n 80 "$STATUS" || true
  fi
  sleep 60
done

