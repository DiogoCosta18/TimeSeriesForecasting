#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/download_from_vast.sh <INSTANCE_ID> <LOCAL_OUTPUT_DIR> [full|incremental|ssh-info]" >&2
  exit 2
fi

INSTANCE_ID="$1"
LOCAL_OUTPUT_DIR="$2"
MODE="${3:-full}"
DOWNLOAD_DEST="$LOCAL_OUTPUT_DIR/downloads"
LOG_DIR="$LOCAL_OUTPUT_DIR/orchestration_logs"
mkdir -p "$DOWNLOAD_DEST" "$LOG_DIR"
DOWNLOAD_LOG="$LOG_DIR/download.log"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$DOWNLOAD_LOG"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" | tee -a "$DOWNLOAD_LOG" >&2
    exit 1
  fi
}

require_command ssh
require_command rsync

if ! command -v vastai >/dev/null 2>&1; then
  python -m pip install -U vastai
fi

resolve_ssh() {
  if [[ -n "${SSH_HOST:-}" && -n "${SSH_PORT:-}" ]]; then
    printf '%s %s\n' "$SSH_HOST" "$SSH_PORT"
    return 0
  fi

  local ssh_url=""
  ssh_url="$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null || true)"
  if [[ -n "$ssh_url" ]]; then
    python - "$ssh_url" <<'PY'
import re, sys
s = sys.argv[1].strip()
patterns = [
    r"ssh://(?:root@)?([^:/\s]+):(\d+)",
    r"(?:root@)?([^:\s]+):(\d+)",
    r"-p\s+(\d+)\s+root@([^\s]+)",
    r"root@([^\s]+)\s+-p\s+(\d+)",
]
for pat in patterns:
    m = re.search(pat, s)
    if not m:
        continue
    if pat.startswith("-p"):
        print(m.group(2), m.group(1))
    else:
        print(m.group(1), m.group(2))
    raise SystemExit(0)
raise SystemExit(1)
PY
    return 0
  fi

  python - "$INSTANCE_ID" <<'PY'
import json, subprocess, sys
instance_id = str(sys.argv[1])
commands = [
    ["vastai", "show", "instance", instance_id, "--raw"],
    ["vastai", "show", "instances", "--raw"],
]
for cmd in commands:
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        payload = json.loads(out)
    except Exception:
        continue
    items = payload if isinstance(payload, list) else [payload] if isinstance(payload, dict) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        ids = {str(item.get(k)) for k in ("id", "instance_id", "machine_id") if item.get(k) is not None}
        if instance_id not in ids and len(items) != 1:
            continue
        host = item.get("ssh_host") or item.get("public_ipaddr") or item.get("host") or item.get("external_ip")
        port = item.get("ssh_port") or item.get("port") or item.get("ssh_port_mapped")
        if host and port:
            print(host, port)
            raise SystemExit(0)
raise SystemExit(1)
PY
}

if ! SSH_INFO="$(resolve_ssh)"; then
  log "Could not resolve SSH host/port for Vast instance $INSTANCE_ID."
  log "Set SSH_HOST and SSH_PORT manually, then rerun:"
  log "SSH_HOST=<host> SSH_PORT=<port> bash scripts/download_from_vast.sh $INSTANCE_ID \"$LOCAL_OUTPUT_DIR\""
  exit 1
fi

SSH_HOST_RESOLVED="$(awk '{print $1}' <<<"$SSH_INFO")"
SSH_PORT_RESOLVED="$(awk '{print $2}' <<<"$SSH_INFO")"
SSH_OPTS="ssh -p $SSH_PORT_RESOLVED -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
REMOTE_OUTPUTS="root@$SSH_HOST_RESOLVED:/workspace/outputs/"

print_recovery() {
  log "SSH command:"
  log "  ssh -p $SSH_PORT_RESOLVED -o StrictHostKeyChecking=no root@$SSH_HOST_RESOLVED"
  log "Manual rsync command:"
  log "  rsync -azP --partial --append-verify -e \"$SSH_OPTS\" $REMOTE_OUTPUTS \"$DOWNLOAD_DEST/\""
  log "Manual destroy command:"
  log "  vastai destroy instance $INSTANCE_ID"
}

if [[ "$MODE" == "ssh-info" ]]; then
  print_recovery
  echo "$SSH_HOST_RESOLVED $SSH_PORT_RESOLVED"
  exit 0
fi

RSYNC_ARGS=(-azP --partial --append-verify)
if [[ "${LOCAL_RSYNC_DELETE:-0}" == "1" ]]; then
  RSYNC_ARGS+=(--delete)
fi

if [[ "$MODE" == "incremental" ]]; then
  RSYNC_ARGS+=(
    --include='*/'
    --include='STATUS.json'
    --include='DONE.json'
    --include='FAILED.json'
    --include='SMOKE_DONE.json'
    --include='SMOKE_FAILED.json'
    --include='FULL_DONE.json'
    --include='FULL_FAILED.json'
    --include='REMOTE_RUNNER_PID.txt'
    --include='run_config.json'
    --include='data_manifest.json'
    --include='smoke.log'
    --include='full_large.log'
    --include='remote_runner.log'
    --include='VAST_HEARTBEAT.log'
    --include='python_version.txt'
    --include='pip_freeze.txt'
    --include='nvidia_smi_start.txt'
    --include='logs/***'
    --include='checkpoints/***'
    --include='reports/***'
    --include='optuna/***'
    --include='final/model_artifacts/***'
    --include='final/forecasts.parquet'
    --include='final/test_metrics.parquet'
    --include='cv/fold_forecasts.parquet'
    --include='cv/fold_metrics.parquet'
    --include='finetuning/finetuned_forecasts.parquet'
    --include='finetuning/*metrics.parquet'
    --exclude='*'
  )
fi

log "Download mode: $MODE"
print_recovery
rsync "${RSYNC_ARGS[@]}" -e "$SSH_OPTS" "$REMOTE_OUTPUTS" "$DOWNLOAD_DEST/" 2>&1 | tee -a "$DOWNLOAD_LOG"

rsync -azP --partial --append-verify -e "$SSH_OPTS" \
  "root@$SSH_HOST_RESOLVED:/workspace/run_forecast_pipeline_vast.sh" \
  "$LOG_DIR/run_forecast_pipeline_vast.sh" 2>/dev/null || true
rsync -azP --partial --append-verify -e "$SSH_OPTS" \
  "root@$SSH_HOST_RESOLVED:/workspace/remote_heartbeat_status.sh" \
  "$LOG_DIR/remote_heartbeat_status.sh" 2>/dev/null || true

python - "$LOCAL_OUTPUT_DIR" "$MODE" "$INSTANCE_ID" <<'PY'
import json, pathlib, sys, datetime
root = pathlib.Path(sys.argv[1])
status = {
    "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "mode": sys.argv[2],
    "instance_id": sys.argv[3],
    "downloads_dir": str(root / "downloads"),
}
(root / "orchestration_logs" / "LOCAL_DOWNLOAD_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
PY

log "Local output path: $DOWNLOAD_DEST"
