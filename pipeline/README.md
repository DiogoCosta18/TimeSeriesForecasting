# Forecasting Pipeline

This repository runs leakage-safe M3/M4 forecasting experiments across six time-series feature regimes, monthly and quarterly frequencies, three STL strategies, four model families, and optional post-global fine-tuning.

The code is designed for interruptible Vast.ai machines:

- deterministic feature sampling manifests
- fold-safe feature computation and STL decomposition
- atomic output writes
- resumable task checkpoints
- Optuna SQLite study paths for auto-model runs
- heartbeat, sync, verification, and opt-in Vast.ai destruction scripts

The execution engine includes production integrations points for `statsforecast`, `mlforecast`, `neuralforecast`, and transformer auto classes. Smoke mode also has deterministic fallback forecasters so the pipeline can validate on machines where heavyweight GPU dependencies are not yet available.

## Local Smoke Test

```bash
bash scripts/run_smoke.sh
```

## Vast.ai Full Run

```bash
VAST_AUTO_DESTROY=1 OUTPUT_SYNC_URI="<my-sync-target>" bash scripts/run_all_vast.sh
```

`VAST_AUTO_DESTROY=1` destroys the Vast.ai instance only after output sync and artifact verification succeed. Auto-destroy is disabled by default.

## CLI

```bash
python -m src.main \
  --config configs/base.yaml \
  --monthly-config configs/monthly.yaml \
  --quarterly-config configs/quarterly.yaml \
  --vast-config configs/vast_gpu.yaml \
  --run-root outputs \
  --budget smoke \
  --resume true
```

Useful filters:

- `--features feature_non_normality,feature_arch_stat`
- `--frequencies monthly`
- `--models AutoETS,AutoRidge,AutoNLinear`
- `--decomposition-methods without_stl,stl_seasonal_naive`
- `--finetuning-modes no_finetune,finetune_by_feature_bucket`

## Budget Modes

- `smoke`: one feature, 20 M3 + 20 M4 per frequency, one window, one model per family, tiny bucket fine-tuning.
- `pilot`: two features, 100 M3 + 100 M4 per frequency, two windows, 10 trials.
- `full`: six features, 750 + 750 per frequency, three windows, 50 trials, bucket fine-tuning enabled, series fine-tuning disabled.
- `large`: six features, 750 + 750 per frequency, three windows, 100 trials, optional series fine-tuning via config.

## Notes

Feature values used for sampling are computed from historical training portions only. Rolling validation recomputes fold-safe features from each fold's training history. STL/MSTL is fitted inside each fold only, never on the full series before validation.

