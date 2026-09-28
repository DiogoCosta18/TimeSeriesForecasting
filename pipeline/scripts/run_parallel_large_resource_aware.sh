#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export TERM=xterm

export FEATURE_COMPUTE_N_JOBS=8

export FORECAST_PARALLEL_ENABLED=1
export FORECAST_CPU_TASK_WORKERS=6
export FORECAST_GPU_TASK_WORKERS=1
export FORECAST_STATISTICAL_WORKERS=4
export FORECAST_MLFORECAST_WORKERS=4
export FORECAST_NEURAL_WORKERS=1
export FORECAST_TRANSFORMER_WORKERS=1
export FORECAST_THREADS_PER_CPU_TASK=1
export FORECAST_THREADS_PER_GPU_TASK=2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m src.main \
  --config configs/generated_large_everything.yaml \
  --monthly-config configs/monthly.yaml \
  --quarterly-config configs/quarterly.yaml \
  --vast-config configs/vast_gpu.yaml \
  --run-root /workspace/outputs_parallel_large_resource_aware \
  --budget large \
  --resume false \
  --finetuning-modes no_finetune,finetune_by_feature_bucket