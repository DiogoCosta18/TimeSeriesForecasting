#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
RUN_ROOT="${RUN_ROOT:-/workspace/outputs}"
mkdir -p "$RUN_ROOT"
if [[ "${OUTPUT_SYNC_URI:-}" == local:* && -z "${VAST_AUTO_DESTROY+x}" ]]; then
  export VAST_AUTO_DESTROY=0
fi

date
hostname
pwd
python --version
python -m pip freeze || true
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "nvidia-smi unavailable"; fi
df -h || true
free -h || true

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip freeze

HEARTBEAT_LOG="$RUN_ROOT/vast_status_$(date +%Y%m%d_%H%M%S).log"
bash scripts/heartbeat_status.sh "$RUN_ROOT" > "$HEARTBEAT_LOG" 2>&1 &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

python -m src.main \
  --config configs/base.yaml \
  --monthly-config configs/monthly.yaml \
  --quarterly-config configs/quarterly.yaml \
  --vast-config configs/vast_gpu.yaml \
  --run-root "$RUN_ROOT" \
  --budget full \
  --resume true

RUN_ROOT="$RUN_ROOT" bash scripts/sync_outputs.sh "$RUN_ROOT"

LATEST_RUN="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'run_*' | sort | tail -n 1 || true)"
VERIFY_OK=1
for required in DONE.json reports/leaderboard_overall.csv reports/summary.md run_config.json data_manifest.json sampling/feature_sample_manifest.parquet sampling/feature_bucket_manifest.parquet; do
  if [[ -z "$LATEST_RUN" || ! -f "$LATEST_RUN/$required" ]]; then
    echo "Verification missing $required"
    VERIFY_OK=0
  fi
done
if [[ -n "${OUTPUT_SYNC_URI:-}" && "${OUTPUT_SYNC_URI:-}" != local:* && ! -f "$RUN_ROOT/SYNC_COMPLETE.json" ]]; then
  echo "Verification missing SYNC_COMPLETE.json"
  VERIFY_OK=0
fi

if [[ "$VERIFY_OK" == "1" ]]; then
  RUN_ROOT="$RUN_ROOT" bash scripts/destroy_instance_if_safe.sh "$RUN_ROOT"
else
  echo "Auto-destroy skipped because outputs were not verified."
fi
