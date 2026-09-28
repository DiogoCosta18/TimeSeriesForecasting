#!/usr/bin/env bash
# Option C: correct stl_model_all_components for global model families.
# Trains one global model per STL component (trend, seasonal, residual) and recomposes.
# Run AFTER the main run completes, with the fixed train_family.py already deployed.
# Uses a separate run root so the existing run's checkpoint is not touched.
set -euo pipefail

export PYTHONUNBUFFERED=1
REPO_DIR="${REPO_DIR:-/workspace/forecast_pipeline}"
VENV_DIR="${VENV_DIR:-/workspace/venv}"
RUN_ROOT="${RUN_ROOT:-/workspace/outputs_optionc}"

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"

mkdir -p "$RUN_ROOT" /workspace/logs

{
  date -u
  echo "=== Option C: stl_model_all_components for global families ==="
  echo "REPO_DIR=$REPO_DIR  RUN_ROOT=$RUN_ROOT"
  python --version || true
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "nvidia-smi unavailable"; fi
} | tee "$RUN_ROOT/optionc_start.log"

# Heartbeat so we can monitor progress remotely
bash "$REPO_DIR/scripts/remote_heartbeat_status.sh" "$RUN_ROOT" &
HEARTBEAT_PID=$!
echo "$HEARTBEAT_PID" > "$RUN_ROOT/HEARTBEAT_PID.txt"
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Launching Option C run" | tee -a "$RUN_ROOT/optionc_start.log"

python -m src.main \
  --config configs/base.yaml \
  --monthly-config configs/monthly.yaml \
  --quarterly-config configs/quarterly.yaml \
  --run-root "$RUN_ROOT" \
  --budget full \
  --resume false \
  --frequencies monthly,quarterly \
  --decomposition-methods stl_model_all_components \
  --models AutoRidge,AutoLinearRegression,AutoXGBoost,AutoRandomForest,AutoNLinear,AutoNHITS,AutoLSTM,AutoTFT,AutoPatchTST \
  2>&1 | tee "$RUN_ROOT/optionc_run.log"

STATUS=${PIPESTATUS[0]}
if [[ "$STATUS" == "0" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Option C DONE" | tee -a "$RUN_ROOT/optionc_run.log"
else
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Option C FAILED with status $STATUS" | tee -a "$RUN_ROOT/optionc_run.log"
  exit "$STATUS"
fi
