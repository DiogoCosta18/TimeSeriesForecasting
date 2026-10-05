#!/usr/bin/env bash
# Prepare a fresh run machine (vast.ai, image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime) the
# same way every time: system tools, the pinned rclone, the pinned environment with its tests
# (D22, I5), and the frozen data copy fetched from the bucket and verified (D2, D20).
#
#   scripts/setup_machine.sh DATA_PARENT
#
# Run from the pipeline folder of a clone of the repository. Needs RERUN_BUCKET and an rclone
# configuration with the remote b2rerun (copied onto the machine by the operator, never stored
# in the repository). The frozen copy ends up in DATA_PARENT/frozen_m3m4_v1.
set -euo pipefail
DATA_PARENT=${1:?usage: setup_machine.sh DATA_PARENT}
: "${RERUN_BUCKET:?set RERUN_BUCKET to the bucket name (never stored in the repository)}"
export PATH="$HOME/.local/bin:$PATH"

missing=()
for tool in git curl rsync tmux python3; do command -v "$tool" >/dev/null || missing+=("$tool"); done
if [ ${#missing[@]} -gt 0 ]; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
fi
[ -x "$HOME/.local/bin/rclone" ] || bash scripts/install_rclone.sh
bash environment/build_env.sh

DATA="$DATA_PARENT/frozen_m3m4_v1"
rclone copy "b2rerun:$RERUN_BUCKET/data/frozen_m3m4_v1" "$DATA" --checksum --log-level NOTICE
rclone check "b2rerun:$RERUN_BUCKET/data/frozen_m3m4_v1" "$DATA" --checksum --one-way --log-level NOTICE
"$HOME/venvs/rerun-v2/bin/python" -m src.cli freeze-data verify --data-dir "$DATA"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "cpus $(nproc), memory $(free -g | awk '/Mem:/ {print $2}') GB, disk $(df -h "$DATA_PARENT" | awk 'NR==2 {print $4}') free"
echo "machine ready: environment $HOME/venvs/rerun-v2, data $DATA"
