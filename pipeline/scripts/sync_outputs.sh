#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-${RUN_ROOT:-/workspace/outputs}}"
DEST="${OUTPUT_SYNC_URI:-}"
export RUN_ROOT

if [[ -z "$DEST" ]]; then
  echo "OUTPUT_SYNC_URI empty; local no-op sync."
elif [[ "$DEST" == local:* ]]; then
  LOCAL_TARGET="${DEST#local:}"
  echo "OUTPUT_SYNC_URI uses local download mode: $LOCAL_TARGET"
  echo "The Vast instance will not push files to your computer."
  echo "Run the local orchestration/download script from your computer so it can pull with rsync over SSH."
  python - <<'PY'
import json, os, pathlib, datetime
root = pathlib.Path(os.environ.get("RUN_ROOT", "/workspace/outputs"))
path = root / "LOCAL_PULL_REQUIRED.json"
path.write_text(json.dumps({"created_at_utc": datetime.datetime.utcnow().isoformat() + "Z", "output_sync_uri": os.environ.get("OUTPUT_SYNC_URI", "")}, indent=2) + "\n")
PY
  exit 0
elif [[ "$DEST" == rclone:* ]]; then
  TARGET="${DEST#rclone:}"
  rclone sync "$RUN_ROOT" "$TARGET"
elif [[ "$DEST" == s3://* ]]; then
  if ! command -v aws >/dev/null 2>&1; then
    echo "aws CLI unavailable for s3 sync" >&2
    exit 1
  fi
  aws s3 sync "$RUN_ROOT" "$DEST"
elif [[ "$DEST" == *"@"*":"* ]]; then
  rsync -az --partial "$RUN_ROOT"/ "$DEST"/
else
  mkdir -p "$DEST"
  cp -a "$RUN_ROOT"/. "$DEST"/
fi

python - <<'PY'
import json, os, pathlib, datetime
root = pathlib.Path(os.environ.get("RUN_ROOT", "/workspace/outputs"))
path = root / "SYNC_COMPLETE.json"
path.write_text(json.dumps({"synced_at_utc": datetime.datetime.utcnow().isoformat() + "Z", "output_sync_uri": os.environ.get("OUTPUT_SYNC_URI", "")}, indent=2) + "\n")
PY
