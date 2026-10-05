"""Evaluation engine (protocol Sections 4.1, 4.4, 4.5, 5.1; D8, D9, D15-D19; tests U13, U16, U17).

One row per (task, series, window) holding the test values, the forecasts and the
seasonal-naive forecasts, every accuracy measure, the keys, the fit and predict times,
the status and the full provenance (D19); STL-AC rows also hold the provenance of each
of their three component models.

- Global models (``evaluate_global_task``): for each window, every member series is cut
  at its own cutoff (window w trains on y[1 .. n - (3 - w)h]); STL is fitted on each
  training window only; one model per target is fitted on the pool from its frozen
  configuration; forecasts are recomposed (STL-SN: T+R forecast plus the last seasonal
  cycle; STL-AC: sum of the T, S and R forecasts). Any exception fails the whole task:
  nothing is returned (D16, U17).
- Statistical models (``evaluate_statistical_series``): local and independent of any
  training pool, so computed once per series, window, strategy and model and shared by
  every feature sample (D18, U16). A per-series exception is recorded as a failed row
  with its type and message and no forecast; for STL-AC any failed component fails the
  row (D16, U13).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.forecast import models
from src.forecast.metrics import row_metrics
from src.forecast.registry import STRATEGY_TARGETS, config_key, family
from src.forecast.result import ForecastResult, forecast_hash
from src.forecast.targets import component_targets, seasonal_continuation
from src.metrics.rel_naive import seasonal_naive_forecast

METRICS = ["relnaive_capped", "relnaive", "mase", "smape", "mae", "pocid"]


@dataclass(frozen=True)
class Provenance:
    """D19: identifies the code, environment, data, manifest and configurations of a row."""
    code_commit: str
    environment_lock_sha256: str
    data_manifest_sha256: str
    bundle_sha256: str
    configs_sha256: str


class GlobalTaskError(RuntimeError):
    """A global task failed; it produced no rows (D16)."""


def _series_arrays(series: pd.DataFrame) -> dict[str, tuple[str, np.ndarray]]:
    out = {}
    for uid, g in series.sort_values(["unique_id", "t"], kind="mergesort").groupby("unique_id", sort=True):
        t = g["t"].to_numpy()
        if not np.array_equal(t, np.arange(1, len(t) + 1)):
            raise ValueError(f"{uid}: positions are not 1..n")
        out[uid] = (g["source_dataset"].iloc[0], g["y"].to_numpy(dtype=float))
    return out


def _window_ends(cutoffs: pd.DataFrame, ids) -> dict[int, dict[str, int]]:
    sub = cutoffs[cutoffs["unique_id"].isin(set(ids))]
    ends = {int(w): dict(zip(g["unique_id"], g["train_end_idx"].astype(int))) for w, g in sub.groupby("window")}
    for w, by_id in ends.items():
        if set(by_id) != set(ids):
            raise ValueError(f"window {w}: cutoffs missing for {len(set(ids) - set(by_id))} series")
    return ends


def _component_record(target: str, res: ForecastResult, trained_steps: int | None = None) -> dict:
    return {"target": target, "config_key": res.config_key, "trained_steps": trained_steps, "backend": res.backend,
            "backend_version": res.backend_version, "device": res.device, "seed": res.seed,
            "forecast_hash": res.forecast_hash}


def _row(keys: dict, uid: str, source: str, window: int, cutoff: int, y_train, y_test, yhat, naive,
         components: list[dict], timing: dict, provenance: Provenance, m: int) -> dict:
    return {
        **keys, "unique_id": uid, "source_dataset": source, "window": window, "cutoff_t": cutoff,
        "status": "trained", "failure": None,
        **row_metrics(y_train, y_test, yhat, naive, m),
        "y_test": [float(v) for v in y_test], "yhat": [float(v) for v in yhat], "yhat_naive": [float(v) for v in naive],
        "forecast_hash": forecast_hash(yhat),
        "backend": components[0]["backend"], "backend_version": components[0]["backend_version"],
        "device": components[0]["device"], "components": json.dumps(components, sort_keys=True),
        "fit_seconds": float(timing["fit_seconds"]), "predict_seconds": float(timing["predict_seconds"]),
        **asdict(provenance),
    }


def evaluate_global_task(task: dict, series: pd.DataFrame, cutoffs: pd.DataFrame, frozen_entries: dict,
                         buckets: dict[str, str], h: int, season_length: int, provenance: Provenance) -> pd.DataFrame:
    """Evaluate one global task on its training pool.

    ``task`` holds feature_name, frequency, strategy, model, scope and seed; ``series`` the
    canonical rows (unique_id, source_dataset, t, y) of exactly the pool's series;
    ``buckets`` maps every series to its bucket for the task's feature (pairing key).
    """
    model, frequency, strategy, seed = task["model"], task["frequency"], task["strategy"], int(task["seed"])
    if family(model) == "statistical":
        raise ValueError("statistical models are evaluated per series (evaluate_statistical_series)")
    keys = {"feature_name": task["feature_name"], "frequency": frequency, "strategy": strategy,
            "family": family(model), "model": model, "scope": task["scope"], "seed": seed}
    arrays = _series_arrays(series)
    ids = sorted(arrays)
    ends = _window_ends(cutoffs, ids)
    entries = {target: frozen_entries[config_key(model, frequency, target)] for target in STRATEGY_TARGETS[strategy]}
    rows = []
    for window in sorted(ends):
        train_end = ends[window]
        parts = {target: [] for target in entries}
        decomposed = {}
        for uid in ids:
            y_train = arrays[uid][1][: train_end[uid]]
            targets = {"raw": y_train} if strategy == "direct" else component_targets(y_train, season_length)
            decomposed[uid] = targets
            for target in entries:
                parts[target].append(pd.DataFrame({"unique_id": uid, "ds": np.arange(1, len(y_train) + 1), "y": targets[target]}))
        results, timing = {}, {"fit_seconds": 0.0, "predict_seconds": 0.0}
        for target, entry in entries.items():
            clock = {}
            results[target] = models.fit_predict_global(
                model, entry["params"], pd.concat(parts[target], ignore_index=True), h, season_length, seed,
                trained_steps=entry.get("trained_steps"), config_key=config_key(model, frequency, target), timings=clock,
            )
            timing = {k: timing[k] + clock[k] for k in timing}
        for uid in ids:
            source, y = arrays[uid]
            y_train, y_test = y[: train_end[uid]], y[train_end[uid]: train_end[uid] + h]
            if len(y_test) != h:
                raise GlobalTaskError(f"{uid}: window {window} has {len(y_test)} test values, expected {h}")
            if strategy == "direct":
                yhat = results["raw"][uid].yhat
            elif strategy == "stl_sn":
                yhat = results["nonseasonal"][uid].yhat + seasonal_continuation(decomposed[uid]["seasonal"], h, season_length)
            else:
                yhat = results["trend"][uid].yhat + results["seasonal"][uid].yhat + results["residual"][uid].yhat
            components = [_component_record(target, results[target][uid], entries[target].get("trained_steps"))
                          for target in entries]
            naive = seasonal_naive_forecast(y_train, h, season_length)
            rows.append(_row({**keys, "bucket": buckets[uid]}, uid, source, window, train_end[uid], y_train, y_test,
                             yhat, naive, components, timing, provenance, season_length))
    return pd.DataFrame(rows)


def _statistical_window(model: str, strategy: str, y_train, h: int, m: int) -> tuple[np.ndarray, list[dict], dict]:
    timing = {"fit_seconds": 0.0, "predict_seconds": 0.0}
    targets = {"raw": np.asarray(y_train, dtype=float)} if strategy == "direct" else component_targets(y_train, m)
    results = {}
    for target in STRATEGY_TARGETS[strategy]:
        clock = {}
        results[target] = models.forecast_statistical(model, targets[target], h, m, timings=clock)
        timing = {k: timing[k] + clock[k] for k in timing}
    if strategy == "direct":
        yhat = results["raw"].yhat
    elif strategy == "stl_sn":
        yhat = results["nonseasonal"].yhat + seasonal_continuation(targets["seasonal"], h, m)
    else:
        yhat = results["trend"].yhat + results["seasonal"].yhat + results["residual"].yhat
    return yhat, [_component_record(t, r) for t, r in results.items()], timing


def evaluate_statistical_series(model: str, frequency: str, strategy: str, series: pd.DataFrame, cutoffs: pd.DataFrame,
                                h: int, season_length: int, provenance: Provenance) -> pd.DataFrame:
    """All windows of one series for one statistical model and strategy (pool-independent, D18)."""
    if family(model) != "statistical":
        raise ValueError(f"{model} is not a statistical model")
    arrays = _series_arrays(series)
    if len(arrays) != 1:
        raise ValueError("evaluate_statistical_series takes exactly one series")
    (uid, (source, y)), = arrays.items()
    keys = {"feature_name": None, "frequency": frequency, "strategy": strategy, "family": "statistical",
            "model": model, "scope": "cohort", "seed": None, "bucket": None}
    rows = []
    for window, train_end in sorted((w, e[uid]) for w, e in _window_ends(cutoffs, [uid]).items()):
        y_train, y_test = y[:train_end], y[train_end: train_end + h]
        try:
            if len(y_test) != h:
                raise ValueError(f"window {window} has {len(y_test)} test values, expected {h}")
            yhat, components, timing = _statistical_window(model, strategy, y_train, h, season_length)
        except Exception as exc:  # D16: a failed row with its reason, no forecast; the task continues
            rows.append({**keys, "unique_id": uid, "source_dataset": source, "window": window, "cutoff_t": train_end,
                         "status": "failed", "failure": f"{type(exc).__name__}: {exc}"[:500],
                         **{metric: np.nan for metric in METRICS}, "y_test": None, "yhat": None, "yhat_naive": None,
                         "forecast_hash": None, "backend": "statsforecast", "backend_version": models.backend_version(model),
                         "device": "cpu", "components": None, "fit_seconds": np.nan, "predict_seconds": np.nan,
                         **asdict(provenance)})
            continue
        naive = seasonal_naive_forecast(y_train, h, season_length)
        rows.append(_row(keys, uid, source, window, train_end, y_train, y_test, yhat, naive, components, timing,
                         provenance, season_length))
    return pd.DataFrame(rows)
