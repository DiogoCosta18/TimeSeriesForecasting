#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-${RUN_ROOT:-/workspace/outputs}}"

if [[ "${OUTPUT_SYNC_URI:-}" == local:* ]]; then
  if [[ "${LOCAL_DOWNLOAD_VERIFIED:-0}" != "1" ]]; then
    echo "Auto-destroy skipped because local OUTPUT_SYNC_URI requires verified local download first."
    echo "Use scripts/vast_launch_train_destroy.sh or scripts/download_from_vast.sh from your local computer."
    exit 0
  fi
fi

if [[ "${VAST_AUTO_DESTROY:-0}" != "1" ]]; then
  echo "Auto-destroy disabled. Set VAST_AUTO_DESTROY=1 to enable."
  exit 0
fi

LATEST_RUN="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'run_*' | sort | tail -n 1 || true)"
if [[ -z "$LATEST_RUN" ]]; then
  echo "Auto-destroy skipped because no run directory was found."
  exit 0
fi

for required in DONE.json reports/leaderboard_overall.csv reports/summary.md run_config.json data_manifest.json sampling/feature_sample_manifest.parquet sampling/feature_bucket_manifest.parquet; do
  if [[ ! -f "$LATEST_RUN/$required" ]]; then
    echo "Auto-destroy skipped because outputs were not verified. Missing $required"
    exit 0
  fi
done

if [[ -n "${OUTPUT_SYNC_URI:-}" && ! -f "$RUN_ROOT/SYNC_COMPLETE.json" ]]; then
  echo "Auto-destroy skipped because outputs were not verified. Missing SYNC_COMPLETE.json"
  exit 0
fi

INSTANCE_ID="${VAST_INSTANCE_ID:-${CONTAINER_ID:-}}"
if [[ -z "$INSTANCE_ID" ]]; then
  echo "Auto-destroy skipped because neither VAST_INSTANCE_ID nor CONTAINER_ID is set."
  exit 0
fi

if ! command -v vastai >/dev/null 2>&1; then
  python -m pip install vastai
fi

echo "All outputs synced and verified. Destroying Vast.ai instance."
vastai destroy instance "$INSTANCE_ID"
