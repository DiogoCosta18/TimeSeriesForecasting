#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
python -m src.main --config configs/base.yaml --monthly-config configs/monthly.yaml --quarterly-config configs/quarterly.yaml --vast-config configs/vast_gpu.yaml --run-root "${RUN_ROOT:-outputs}" --budget "${BUDGET:-full}" --resume true

