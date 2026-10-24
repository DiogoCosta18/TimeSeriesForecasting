# Inventory of the frozen run full3_20261011 for the revision analyses

Checked on 2026-10-24 against the run folder, its sensitivity studies (S3, S4, S4s, S5) and the pipeline code.

| # | Artefact | Status | Where / what follows |
|---|---|---|---|
| I1 | MAE of the model and of the seasonal naive per forecast, every frozen configuration | exists | `mae` and `relnaive` in every row (MAE of the seasonal naive = `mae / relnaive`): run (Direct, STL-SN, STL-AC default; cohort and terciles; three seeds), S3 (season length 1), S4 (STL-SN periodic, span 25, re-tuned), S4s (periodic, extra seeds), S5 (random groups, a third of the steps). In `fc_errors` (6,538,743 forecasts). |
| I2 | Per-step forecasts (`y_hat`, `y_true`) | exists | `y_test`, `yhat`, `yhat_naive` in every task parquet (the merged table leaves them out). `fc_steps` is built from them when an analysis needs it. |
| I3 | Component forecasts of the default STL-AC (`t_hat`, `s_hat`, `r_hat`) | missing | `components` holds only each component's configuration and forecast hash. Variant (ii) is run for the periodic and span-25 settings only. |
| I4 | AutoETS specification chosen per series and window | missing | Not stored. AutoETS is refitted under Direct and checked against the stored forecasts. |
| I5 | Frozen configurations of every study | exists | `configs_frozen.json` of the run (90 entries: all hyperparameters, `trained_steps`, input size, patch) and of each study. |
| I6 | Re-tuning of XGBoost, NHITS and PatchTST on the periodic target | exists | `sensitivity_s4/tune` (Optuna studies) and the frozen entries of S4. |
| I7 | — | not needed | — |
| I8 | Dated logs of the main run and of S3–S5, S4s, E10 | exists | `logs/` and the `completed_at_utc` and `compute_seconds` of every task record. |
| I9 | Environment and hardware | exists | `environment/requirements-lock-linux-cu121.txt` (Python 3.11.10, torch 2.5.1+cu121); the platform of every task (AMD EPYC 7B13, RTX 4090, pinned BLAS threads) in its record. |
| I10 | Exact STL calls (default, periodic, span 25) | exists | `src/forecast/targets.py` and `src/forecast/mstl_features.py`. |
| I11 | M3 and M4 data, sampled IDs, start dates | exists | Frozen copy (manifest `m3m4_v1`); IDs and selection seed in `prepare/samples.parquet`; start dates: M3 in the frozen copy (`ds_source`), M4 in the raw `M4-info.csv`. |
| I12 | Per-forecast errors of the extra seeds | exists | Run rows of seeds 20260522 and 20260523 (A15) and S4s. |
| I13 | Errors of the third-of-steps variants | exists | S5 scopes `*_s3` (`tercile_third`, `random_third` in `fc_errors`). |
| I14 | Partial Spearman, Jonckheere–Terpstra, Table 6 bootstrap | exists | `src/analysis/review.py` (`partial_rho`, `partial_ci`), `src/analysis/stats.py` (`jonckheere_terpstra`). |

## Canonical tables

Built by `python -m src.analysis.v2.tables` into a folder outside the repository (large files):

- `fc_errors/` per source: run 2,864,511; S3 101,916; S4 1,009,116; S4s 100,800; S5 2,462,400 forecasts.
- `series_meta.parquet`: 8,493 series (length, history, six features, feature samples, ϕ7, ϕ′4). The start dates are added for E4.
- `inst.parquet`: 14,400 instances (series × feature sample).

## A0

`A0_check.csv`: 118 published values recomputed from the canonical tables (Table 3, rows 1–3; Table 4; Table 8 and the per-model and re-tuning values; Table 11; Table 12; Table C7; `forecasts-total`), none differs.
