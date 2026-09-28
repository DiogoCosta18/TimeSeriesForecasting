#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERE="$(pwd)"
if [[ -f "$HERE/pyproject.toml" && -d "$HERE/src" ]]; then
  REPO_DIR="$HERE"
elif [[ -f "$SCRIPT_DIR/../pyproject.toml" ]]; then
  REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -d "$HERE/forecast_pipeline" ]]; then
  REPO_DIR="$(cd "$HERE/forecast_pipeline" && pwd)"
elif [[ -d "$HERE/forecasting_pipeline" ]]; then
  REPO_DIR="$(cd "$HERE/forecasting_pipeline" && pwd)"
else
  echo "Could not find forecast_pipeline/forecasting_pipeline. Run from the repo or its parent." >&2
  exit 1
fi
cd "$REPO_DIR"

timestamp() { date -u +%Y%m%d_%H%M%S; }
log_ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

if [[ -z "${VAST_API_KEY:-}" ]]; then
  echo "VAST_API_KEY is required. Export it before running this script." >&2
  exit 1
fi

for cmd in python ssh rsync; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required local command not found: $cmd" >&2
    exit 1
  fi
done

if ! command -v vastai >/dev/null 2>&1; then
  python -m pip install -U vastai
fi
vastai set api-key "$VAST_API_KEY" >/dev/null

ensure_ssh_key() {
  mkdir -p "$HOME/.ssh"
  chmod 700 "$HOME/.ssh"
  if [[ ! -f "$HOME/.ssh/id_ed25519.pub" ]]; then
    log "Creating local SSH key at ~/.ssh/id_ed25519 for Vast.ai access."
    ssh-keygen -t ed25519 -C "forecast-pipeline-vast-$(hostname)-$(timestamp)" -f "$HOME/.ssh/id_ed25519" -N ""
  fi
  chmod 600 "$HOME/.ssh/id_ed25519" 2>/dev/null || true
  chmod 644 "$HOME/.ssh/id_ed25519.pub" 2>/dev/null || true
  log "Registering local SSH public key with Vast.ai account if needed."
  vastai create ssh-key "$(cat "$HOME/.ssh/id_ed25519.pub")" -y >/dev/null 2>&1 || true
}

attach_ssh_key_to_instance() {
  local instance_id="$1"
  if [[ -f "$HOME/.ssh/id_ed25519.pub" ]]; then
    log "Attaching local SSH public key to Vast instance $instance_id."
    vastai attach ssh "$instance_id" "$HOME/.ssh/id_ed25519.pub" >/dev/null 2>&1 || \
      vastai attach ssh "$instance_id" "$(cat "$HOME/.ssh/id_ed25519.pub")" >/dev/null 2>&1 || true
  fi
}

if [[ -z "${OUTPUT_SYNC_URI:-}" ]]; then
  LOCAL_OUTPUT_DIR="./vast_downloads/forecast_large_$(timestamp)"
  export OUTPUT_SYNC_URI="local:$LOCAL_OUTPUT_DIR"
  echo "OUTPUT_SYNC_URI was empty; using $OUTPUT_SYNC_URI"
elif [[ "$OUTPUT_SYNC_URI" == local:* ]]; then
  LOCAL_OUTPUT_DIR="${OUTPUT_SYNC_URI#local:}"
else
  echo "Default sync mode is local. OUTPUT_SYNC_URI must be empty or local:<path>." >&2
  exit 1
fi

if [[ -z "${VAST_AUTO_DESTROY+x}" ]]; then
  export VAST_AUTO_DESTROY=0
fi

mkdir -p "$LOCAL_OUTPUT_DIR/orchestration_logs" "$LOCAL_OUTPUT_DIR/downloads" "$LOCAL_OUTPUT_DIR/status_snapshots"
LOG_DIR="$LOCAL_OUTPUT_DIR/orchestration_logs"
LAUNCH_LOG="$LOG_DIR/launch.log"
UPLOAD_LOG="$LOG_DIR/upload.log"
VERIFY_LOG="$LOG_DIR/download_verify.log"
DESTROY_LOG="$LOG_DIR/destroy.log"
DOWNLOAD_INTERVAL_SECONDS="${DOWNLOAD_INTERVAL_SECONDS:-1800}"
VAST_MAX_PRICE="${VAST_MAX_PRICE:-0.25}"
VAST_MIN_RELIABILITY="${VAST_MIN_RELIABILITY:-0.99}"
VAST_DISK_GB="${VAST_DISK_GB:-500}"
VAST_MIN_GPU_RAM="${VAST_MIN_GPU_RAM:-23}"
VAST_GPU_NAME="${VAST_GPU_NAME:-}"
VAST_IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
LABEL="forecast-pipeline-large-$(timestamp)"

log() {
  echo "[$(log_ts)] $*" | tee -a "$LAUNCH_LOG" >&2
}

manual_commands() {
  local instance_id="${1:-<INSTANCE_ID>}"
  local host="${2:-\$SSH_HOST}"
  local port="${3:-\$SSH_PORT}"
  {
    echo "SSH command:"
    echo "  ssh -p $port -o StrictHostKeyChecking=no root@$host"
    echo "Manual rsync command:"
    echo "  rsync -azP --partial --append-verify -e \"ssh -p $port -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10\" root@$host:/workspace/outputs/ \"$LOCAL_OUTPUT_DIR/downloads/\""
    echo "Manual download command:"
    echo "  bash scripts/download_from_vast.sh $instance_id \"$LOCAL_OUTPUT_DIR\""
    echo "Manual destroy command:"
    echo "  vastai destroy instance $instance_id"
    echo "Remote resume command:"
    echo "  cd /workspace/forecast_pipeline && source /workspace/venv/bin/activate && python -m src.main --config configs/generated_large_everything.yaml --monthly-config configs/monthly.yaml --quarterly-config configs/quarterly.yaml --vast-config configs/vast_gpu.yaml --run-root /workspace/outputs --budget large --resume true"
  } | tee -a "$LAUNCH_LOG"
}

select_offer() {
  local query="$1"
  local raw_file="$2"
  log "Searching Vast offers: $query"
  if ! vastai search offers "$query" --raw > "$raw_file"; then
    return 1
  fi
  python - "$raw_file" "$LOG_DIR/selected_offer.json" "$VAST_MAX_PRICE" "$VAST_DISK_GB" "$VAST_MIN_GPU_RAM" "$VAST_GPU_NAME" <<'PY'
import json, math, pathlib, sys
raw_path = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
max_price = float(sys.argv[3])
min_disk = float(sys.argv[4])
min_gpu_ram = float(sys.argv[5])
gpu_name_filter = sys.argv[6].strip().lower()
try:
    data = json.loads(raw_path.read_text())
except Exception:
    raise SystemExit(2)
offers = data if isinstance(data, list) else data.get("offers", []) if isinstance(data, dict) else []
def num(o, *names, default=0.0):
    for name in names:
        try:
            v = o.get(name)
            if v is not None:
                return float(v)
        except Exception:
            pass
    return default
def truth(o, *names):
    for name in names:
        v = o.get(name)
        if isinstance(v, bool):
            return v
        if str(v).lower() in {"true", "1", "yes"}:
            return True
    return False
def score(o):
    reliability = num(o, "reliability", "reliability2", default=0.0)
    dlperf = num(o, "dlperf", "dlperf_per_dphtotal", default=0.0)
    price = num(o, "dph_total", "price", default=999.0)
    value = dlperf / max(price, 1e-9)
    disk_bw = num(o, "disk_bw", "disk_io", "read_bw", "write_bw", default=0.0)
    inet = num(o, "inet_down", default=0.0) + num(o, "inet_up", default=0.0)
    location = " ".join(str(o.get(k, "")) for k in ("geolocation", "country", "location")).lower()
    eu_bonus = 1 if any(x in location for x in ["germany", "de", "france", "netherlands", "spain", "portugal", "europe", "eu"]) else 0
    return (
        1 if truth(o, "verified") else 0,
        value,
        dlperf,
        -price,
        reliability,
        disk_bw,
        inet,
        eu_bonus,
    )
valid = []
for offer in offers:
    if not isinstance(offer, dict):
        continue
    name = str(offer.get("gpu_name", offer.get("gpu_name_id", ""))).lower()
    if gpu_name_filter and gpu_name_filter not in name:
        continue
    gpu_ram = num(offer, "gpu_ram", "gpu_total_ram", default=0.0)
    if gpu_ram and gpu_ram < min_gpu_ram:
        continue
    price = num(offer, "dph_total", "total_dph", "price", default=999.0)
    disk = num(offer, "disk_space", "disk_total", "storage_total", default=0.0)
    if price > max_price:
        continue
    if disk and disk < min_disk:
        continue
    valid.append(offer)
if not valid:
    raise SystemExit(3)
best = sorted(valid, key=score, reverse=True)[0]
out_path.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
print(best.get("id") or best.get("ask_contract_id") or best.get("bundle_id"))
PY
}

find_offer() {
  local search_dir="$LOG_DIR/offer_searches"
  mkdir -p "$search_dir"
  local gpu_filter=""
  if [[ -n "$VAST_GPU_NAME" ]]; then
    gpu_filter="gpu_name=$VAST_GPU_NAME "
  fi
  local queries=(
    "${gpu_filter}num_gpus=1 gpu_ram>=$VAST_MIN_GPU_RAM disk_space>=$VAST_DISK_GB reliability>=$VAST_MIN_RELIABILITY verified=true rentable=true inet_down>=100 inet_up>=50 dph_total<=$VAST_MAX_PRICE"
    "${gpu_filter}num_gpus=1 gpu_ram>=$VAST_MIN_GPU_RAM disk_space>=$VAST_DISK_GB reliability>=0.985 verified=true rentable=true inet_down>=100 inet_up>=50 dph_total<=$VAST_MAX_PRICE"
    "${gpu_filter}num_gpus=1 gpu_ram>=$VAST_MIN_GPU_RAM disk_space>=$VAST_DISK_GB reliability>=0.985 verified=true rentable=true dph_total<=$VAST_MAX_PRICE"
    "${gpu_filter}num_gpus=1 gpu_ram>=$VAST_MIN_GPU_RAM disk_space>=$VAST_DISK_GB verified=true rentable=true dph_total<=$VAST_MAX_PRICE"
    "${gpu_filter}num_gpus=1 gpu_ram>=$VAST_MIN_GPU_RAM verified=true rentable=true dph_total<=$VAST_MAX_PRICE"
  )
  local i=0
  for query in "${queries[@]}"; do
    i=$((i + 1))
    local raw_file="$search_dir/offers_$i.json"
    if OFFER_ID="$(select_offer "$query" "$raw_file")"; then
      if [[ -n "$OFFER_ID" && "$OFFER_ID" != "None" ]]; then
        echo "$OFFER_ID"
        return 0
      fi
    fi
  done
  return 1
}

create_instance() {
  local offer_id="$1"
  log "Creating Vast instance from offer $offer_id with disk ${VAST_DISK_GB}GB and image $VAST_IMAGE"
  local create_log="$LOG_DIR/create_instance_output.txt"
  set +e
  vastai create instance "$offer_id" \
    --image "$VAST_IMAGE" \
    --disk "$VAST_DISK_GB" \
    --ssh \
    --direct \
    --label "$LABEL" \
    --cancel-unavail > >(tee "$create_log" >&2) 2> >(tee -a "$create_log" >&2)
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" != "0" ]]; then
    log "Instance creation failed. See $create_log"
    return "$status"
  fi
  python - "$create_log" "$LOG_DIR/instance.json" <<'PY'
import json, pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text(errors="replace")
instance_id = None
try:
    payload = json.loads(text)
    if isinstance(payload, dict):
        for key in ("new_contract", "instance_id", "id", "contract_id"):
            if payload.get(key):
                instance_id = str(payload[key])
                break
except Exception:
    pass
if instance_id is None:
    patterns = [
        r"instance\s+(\d+)",
        r"new_contract[^\d]*(\d+)",
        r"contract[^\d]*(\d+)",
        r"\b(\d{5,})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            instance_id = m.group(1)
            break
if not instance_id:
    raise SystemExit(1)
pathlib.Path(sys.argv[2]).write_text(json.dumps({"instance_id": instance_id, "raw_create_output": text}, indent=2) + "\n")
print(instance_id)
PY
}

wait_for_ssh() {
  local instance_id="$1"
  local deadline=$((SECONDS + 2700))
  log "Waiting for instance $instance_id SSH availability, timeout 45 minutes"
  while (( SECONDS < deadline )); do
    if SSH_INFO="$(bash scripts/download_from_vast.sh "$instance_id" "$LOCAL_OUTPUT_DIR" ssh-info 2>/dev/null | tail -n 1)" && [[ "$SSH_INFO" =~ ^[^[:space:]]+[[:space:]][0-9]+$ ]]; then
      SSH_HOST="$(awk '{print $1}' <<<"$SSH_INFO")"
      SSH_PORT="$(awk '{print $2}' <<<"$SSH_INFO")"
      if ssh -p "$SSH_PORT" \
        -o BatchMode=yes \
        -o IdentitiesOnly=yes \
        -o ConnectTimeout=10 \
        -o StrictHostKeyChecking=no \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=10 \
        "root@$SSH_HOST" "true" >/dev/null 2>&1; then
        {
          echo "INSTANCE_ID=$instance_id"
          echo "SSH_HOST=$SSH_HOST"
          echo "SSH_PORT=$SSH_PORT"
          echo "SSH_COMMAND=ssh -p $SSH_PORT -o StrictHostKeyChecking=no root@$SSH_HOST"
          echo "MANUAL_RSYNC=rsync -azP --partial --append-verify -e \"ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10\" root@$SSH_HOST:/workspace/outputs/ \"$LOCAL_OUTPUT_DIR/downloads/\""
          echo "MANUAL_DESTROY=vastai destroy instance $instance_id"
        } > "$LOG_DIR/ssh_info.txt"
        log "SSH is ready: root@$SSH_HOST:$SSH_PORT"
        return 0
      fi
      log "SSH endpoint exists at root@$SSH_HOST:$SSH_PORT but is not accepting connections yet."
    fi
    sleep 30
  done
  log "Timed out waiting for SSH."
  return 1
}

remote_ssh() {
  ssh -p "$SSH_PORT" \
    -o BatchMode=yes \
    -o IdentitiesOnly=yes \
    -o ConnectTimeout=30 \
    -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=10 \
    "root@$SSH_HOST" "$@"
}

upload_repo() {
  log "Uploading repository $REPO_DIR to /workspace/forecast_pipeline"
  local mkdir_ok=0
  for attempt in 1 2 3 4 5; do
    if remote_ssh "mkdir -p /workspace/forecast_pipeline /workspace/outputs /workspace/logs"; then
      mkdir_ok=1
      break
    fi
    log "Remote mkdir failed on attempt $attempt; waiting before retry."
    sleep 20
  done
  if [[ "$mkdir_ok" != "1" ]]; then
    log "Could not create remote workspace directories over SSH."
    return 1
  fi
  set +e
  for attempt in 1 2 3; do
    rsync -azP --delete \
    --exclude='outputs/' \
    --exclude='vast_downloads/' \
    --exclude='vast_orchestration_logs/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10" \
    "$REPO_DIR/" "root@$SSH_HOST:/workspace/forecast_pipeline/" 2>&1 | tee -a "$UPLOAD_LOG"
    local status=${PIPESTATUS[0]}
    if [[ "$status" == "0" ]]; then
      break
    fi
    log "rsync upload failed on attempt $attempt; waiting before retry."
    sleep 30
  done
  set -e
  if [[ "$status" != "0" ]]; then
    log "rsync upload failed; trying tar over SSH fallback."
    tar -C "$REPO_DIR" \
      --exclude='./outputs' \
      --exclude='./vast_downloads' \
      --exclude='./vast_orchestration_logs' \
      --exclude='./.venv' \
      -czf - . | ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no "root@$SSH_HOST" "mkdir -p /workspace/forecast_pipeline && tar -xzf - -C /workspace/forecast_pipeline" 2>&1 | tee -a "$UPLOAD_LOG"
  fi
  remote_ssh "cp /workspace/forecast_pipeline/scripts/remote_run_forecast_pipeline_vast.sh /workspace/run_forecast_pipeline_vast.sh && cp /workspace/forecast_pipeline/scripts/remote_heartbeat_status.sh /workspace/remote_heartbeat_status.sh && chmod +x /workspace/run_forecast_pipeline_vast.sh /workspace/remote_heartbeat_status.sh"
}

start_remote_runner() {
  log "Starting remote runner in tmux/screen/nohup"
  remote_ssh "if command -v tmux >/dev/null 2>&1; then if ! tmux has-session -t forecast_large 2>/dev/null; then tmux new-session -d -s forecast_large 'bash /workspace/run_forecast_pipeline_vast.sh'; fi; echo tmux > /workspace/outputs/REMOTE_RUNNER_PID.txt; elif command -v screen >/dev/null 2>&1; then screen -dmS forecast_large bash /workspace/run_forecast_pipeline_vast.sh; echo screen > /workspace/outputs/REMOTE_RUNNER_PID.txt; else nohup bash /workspace/run_forecast_pipeline_vast.sh > /workspace/outputs/remote_runner.log 2>&1 & echo \$! > /workspace/outputs/REMOTE_RUNNER_PID.txt; fi"
}

remote_job_running() {
  remote_ssh "if command -v tmux >/dev/null 2>&1 && tmux has-session -t forecast_large 2>/dev/null; then exit 0; fi; if command -v screen >/dev/null 2>&1 && screen -list | grep -q forecast_large; then exit 0; fi; if [[ -f /workspace/outputs/REMOTE_RUNNER_PID.txt ]]; then pid=\$(cat /workspace/outputs/REMOTE_RUNNER_PID.txt); [[ \"\$pid\" =~ ^[0-9]+$ ]] && kill -0 \"\$pid\" 2>/dev/null && exit 0; fi; exit 1" >/dev/null 2>&1
}

download_outputs() {
  local mode="${1:-incremental}"
  bash scripts/download_from_vast.sh "$INSTANCE_ID" "$LOCAL_OUTPUT_DIR" "$mode"
}

print_local_status() {
  local status_file
  status_file="$(find "$LOCAL_OUTPUT_DIR/downloads" -name STATUS.json -print 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "$status_file" ]]; then
    cp "$status_file" "$LOCAL_OUTPUT_DIR/status_snapshots/STATUS_$(timestamp).json" || true
    python - "$status_file" <<'PY' || true
import json, sys
data = json.load(open(sys.argv[1]))
print("Progress:", data.get("percent_complete_estimate"), "%", "stage:", data.get("current_stage"), "completed:", data.get("completed_tasks"), "failed:", data.get("failed_tasks"))
PY
  fi
}

verify_local_download() {
  local downloads="$LOCAL_OUTPUT_DIR/downloads"
  local latest_run
  latest_run="$(find "$downloads" -mindepth 1 -maxdepth 1 -type d -name 'run_*' | sort | tail -n 1 || true)"
  {
    echo "Verifying local download at $downloads"
    local ok=1
    if [[ -z "$latest_run" ]]; then
      echo "Missing full run directory under $downloads"
      ok=0
    else
      echo "Latest full run: $latest_run"
      for rel in DONE.json run_config.json data_manifest.json STATUS.json reports/leaderboard_overall.csv reports/summary.md; do
        if [[ ! -s "$latest_run/$rel" ]]; then
          echo "Missing or zero-size: $latest_run/$rel"
          ok=0
        else
          echo "OK: $latest_run/$rel ($(wc -c < "$latest_run/$rel") bytes)"
        fi
      done
      for dir in final/model_artifacts checkpoints; do
        if [[ ! -d "$latest_run/$dir" ]]; then
          echo "Missing directory: $latest_run/$dir"
          ok=0
        else
          echo "OK directory: $latest_run/$dir"
        fi
      done
      if [[ -f "$latest_run/FAILED.json" ]]; then
        echo "Found FAILED.json in full run: $latest_run/FAILED.json"
        ok=0
      fi
    fi
    for rel in smoke.log full_large.log; do
      if [[ ! -s "$downloads/$rel" ]]; then
        echo "Missing or zero-size: $downloads/$rel"
        ok=0
      else
        echo "OK: $downloads/$rel ($(wc -c < "$downloads/$rel") bytes)"
      fi
    done
    if [[ ! -s "$LOCAL_OUTPUT_DIR/LOCAL_DOWNLOAD_COMPLETE.json" ]]; then
      echo "Missing LOCAL_DOWNLOAD_COMPLETE.json"
      ok=0
    fi
    [[ "$ok" == "1" ]]
  } | tee -a "$VERIFY_LOG"
}

write_download_complete() {
  python - "$LOCAL_OUTPUT_DIR" "$INSTANCE_ID" <<'PY'
import json, pathlib, sys, datetime
root = pathlib.Path(sys.argv[1])
payload = {
    "completed_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "instance_id": sys.argv[2],
    "downloads_dir": str(root / "downloads"),
}
(root / "LOCAL_DOWNLOAD_COMPLETE.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
}

destroy_if_safe() {
  if [[ "${VAST_AUTO_DESTROY:-0}" != "1" ]]; then
    {
      echo "Instance preserved because VAST_AUTO_DESTROY=0."
      echo "Manual download command: bash scripts/download_from_vast.sh $INSTANCE_ID \"$LOCAL_OUTPUT_DIR\""
      echo "Manual destroy command: vastai destroy instance $INSTANCE_ID"
    } | tee -a "$DESTROY_LOG"
    return 0
  fi
  {
    echo "All outputs downloaded and verified locally. Destroying Vast.ai instance $INSTANCE_ID."
    vastai destroy instance "$INSTANCE_ID"
  } 2>&1 | tee -a "$DESTROY_LOG" || {
    echo "Destroy failed. Manual cleanup: vastai destroy instance $INSTANCE_ID" | tee -a "$DESTROY_LOG"
    return 1
  }
}

trap 'echo "Interrupted. Remote job should continue if it was started."; manual_commands "${INSTANCE_ID:-<INSTANCE_ID>}" "${SSH_HOST:-\$SSH_HOST}" "${SSH_PORT:-\$SSH_PORT}"' INT TERM

log "Repository: $REPO_DIR"
log "Local output directory: $LOCAL_OUTPUT_DIR"
log "OUTPUT_SYNC_URI: $OUTPUT_SYNC_URI"
log "VAST_AUTO_DESTROY: $VAST_AUTO_DESTROY"
ensure_ssh_key

if [[ -n "${VAST_INSTANCE_ID:-}" ]]; then
  INSTANCE_ID="$VAST_INSTANCE_ID"
  log "Using existing instance $INSTANCE_ID"
else
  OFFER_ID="$(find_offer)" || {
    log "No suitable RTX 3090 Vast offer found."
    exit 1
  }
  log "Selected offer $OFFER_ID. Details: $LOG_DIR/selected_offer.json"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "Dry run complete. No instance was created."
    exit 0
  fi
  INSTANCE_ID="$(create_instance "$OFFER_ID")"
  export VAST_INSTANCE_ID="$INSTANCE_ID"
  log "Created instance $INSTANCE_ID"
fi
attach_ssh_key_to_instance "$INSTANCE_ID"

if [[ "$DRY_RUN" == "1" ]]; then
  log "Dry run complete for existing instance $INSTANCE_ID. No upload or training started."
  exit 0
fi

wait_for_ssh "$INSTANCE_ID"
manual_commands "$INSTANCE_ID" "$SSH_HOST" "$SSH_PORT"
upload_repo
start_remote_runner

last_download=0
smoke_pulled=0
log "Monitoring remote job. Incremental downloads every $DOWNLOAD_INTERVAL_SECONDS seconds."
while remote_job_running; do
  now="$SECONDS"
  if remote_ssh "test -f /workspace/outputs/SMOKE_DONE.json -o -f /workspace/outputs/SMOKE_FAILED.json" >/dev/null 2>&1 && [[ "$smoke_pulled" == "0" ]]; then
    log "Smoke marker found. Pulling outputs immediately."
    download_outputs incremental || true
    print_local_status
    smoke_pulled=1
    last_download="$now"
  elif (( now - last_download >= DOWNLOAD_INTERVAL_SECONDS )); then
    log "Periodic incremental download."
    download_outputs incremental || true
    print_local_status
    last_download="$now"
  else
    print_local_status
  fi
  sleep 60
done

log "Remote runner is no longer active. Pulling final outputs."
download_outputs full || {
  log "Final download failed. Instance will not be destroyed."
  manual_commands "$INSTANCE_ID" "$SSH_HOST" "$SSH_PORT"
  exit 1
}
write_download_complete

if [[ -f "$LOCAL_OUTPUT_DIR/downloads/SMOKE_FAILED.json" ]]; then
  log "Smoke failed. Full run was not started. Instance will not be destroyed automatically."
  manual_commands "$INSTANCE_ID" "$SSH_HOST" "$SSH_PORT"
  exit 10
fi

if [[ -f "$LOCAL_OUTPUT_DIR/downloads/FULL_FAILED.json" ]]; then
  log "Full large run failed. Instance will be preserved for inspection/resume."
  manual_commands "$INSTANCE_ID" "$SSH_HOST" "$SSH_PORT"
  exit 20
fi

if verify_local_download; then
  destroy_if_safe
else
  log "Local download verification failed. Instance will not be destroyed."
  manual_commands "$INSTANCE_ID" "$SSH_HOST" "$SSH_PORT"
  exit 1
fi
