#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
RUN_ROOT="${RUN_ROOT:-/workspace/outputs}"
REPO_DIR="${REPO_DIR:-/workspace/forecast_pipeline}"
VENV_DIR="${VENV_DIR:-/workspace/venv}"
if [[ -z "${FEATURE_COMPUTE_N_JOBS:-}" ]]; then
  export FEATURE_COMPUTE_N_JOBS="$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))"
fi
mkdir -p "$RUN_ROOT" /workspace/logs

cd "$REPO_DIR"

{
  date -u
  hostname
  pwd
  echo "FEATURE_COMPUTE_N_JOBS=$FEATURE_COMPUTE_N_JOBS"
  python --version || true
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "nvidia-smi unavailable"; fi
  df -h || true
  free -h || true
} | tee "$RUN_ROOT/remote_system_info_start.log"

if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update || true
  apt-get install -y git rsync screen tmux curl || true
fi

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . || true

python --version > "$RUN_ROOT/python_version.txt"
python -m pip freeze > "$RUN_ROOT/pip_freeze.txt"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi > "$RUN_ROOT/nvidia_smi_start.txt" || true; fi

cp "$REPO_DIR/scripts/remote_heartbeat_status.sh" /workspace/remote_heartbeat_status.sh
chmod +x /workspace/remote_heartbeat_status.sh
bash /workspace/remote_heartbeat_status.sh "$RUN_ROOT" &
HEARTBEAT_PID=$!
echo "$HEARTBEAT_PID" > "$RUN_ROOT/REMOTE_HEARTBEAT_PID.txt"
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

echo "Starting smoke run at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$RUN_ROOT/smoke.log"
set +e
python -m pytest 2>&1 | tee -a "$RUN_ROOT/smoke.log"
PYTEST_STATUS=${PIPESTATUS[0]}
python -m src.main \
  --config configs/base.yaml \
  --monthly-config configs/monthly.yaml \
  --quarterly-config configs/quarterly.yaml \
  --vast-config configs/vast_gpu.yaml \
  --run-root "$RUN_ROOT/smoke" \
  --budget smoke \
  --resume true \
  --debug true \
  2>&1 | tee -a "$RUN_ROOT/smoke.log"
SMOKE_STATUS=${PIPESTATUS[0]}
set -e

SMOKE_DONE="$(find "$RUN_ROOT/smoke" -name DONE.json -print 2>/dev/null | sort | tail -n 1 || true)"
SMOKE_REPORT="$(find "$RUN_ROOT/smoke" -path '*/reports/summary.md' -print 2>/dev/null | sort | tail -n 1 || true)"
if [[ "$PYTEST_STATUS" != "0" || "$SMOKE_STATUS" != "0" || -z "$SMOKE_DONE" || -z "$SMOKE_REPORT" ]]; then
  python - "$RUN_ROOT" "$PYTEST_STATUS" "$SMOKE_STATUS" <<'PY'
import json, pathlib, sys, datetime
root = pathlib.Path(sys.argv[1])
payload = {
    "failed_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "pytest_status": int(sys.argv[2]),
    "smoke_status": int(sys.argv[3]),
}
(root / "SMOKE_FAILED.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
  echo "Smoke failed. Full large run will not start." | tee -a "$RUN_ROOT/smoke.log"
  exit 10
fi

python - "$RUN_ROOT" <<'PY'
import json, pathlib, datetime, sys
root = pathlib.Path(sys.argv[1])
(root / "SMOKE_DONE.json").write_text(json.dumps({"finished_at_utc": datetime.datetime.utcnow().isoformat() + "Z"}, indent=2) + "\n")
PY

echo "Starting full large run at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$RUN_ROOT/full_large.log"
set +e
python -m src.main \
  --config configs/generated_large_everything.yaml \
  --monthly-config configs/monthly.yaml \
  --quarterly-config configs/quarterly.yaml \
  --vast-config configs/vast_gpu.yaml \
  --run-root "$RUN_ROOT" \
  --budget large \
  --resume true \
  2>&1 | tee -a "$RUN_ROOT/full_large.log"
FULL_STATUS=${PIPESTATUS[0]}
set -e

if [[ "$FULL_STATUS" == "0" ]]; then
  python - "$RUN_ROOT" <<'PY'
import json, pathlib, datetime, sys
root = pathlib.Path(sys.argv[1])
(root / "FULL_DONE.json").write_text(json.dumps({"finished_at_utc": datetime.datetime.utcnow().isoformat() + "Z"}, indent=2) + "\n")
PY
else
  python - "$RUN_ROOT" "$FULL_STATUS" <<'PY'
import json, pathlib, datetime, sys
root = pathlib.Path(sys.argv[1])
(root / "FULL_FAILED.json").write_text(json.dumps({"failed_at_utc": datetime.datetime.utcnow().isoformat() + "Z", "status": int(sys.argv[2])}, indent=2) + "\n")
PY
fi

exit "$FULL_STATUS"
