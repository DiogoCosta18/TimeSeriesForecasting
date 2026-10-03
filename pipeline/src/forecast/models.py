"""Forecasts from frozen configurations (protocol Sections 4.2-4.5; tests U10, U11).

- Statistical models (D11): fitted per series, specification chosen by information
  criteria; a failure raises (the engine records a failed row, D16).
- Global models: one model fitted on all series of a training pool from its frozen
  configuration, then h forecasts per series; any failure raises and fails the whole task.
Global libraries receive the 1-based positions t as time stamps (freq = 1), so the
models see only the order of observations.
"""
from __future__ import annotations

import logging
import os
import time
import warnings
from importlib import metadata

import numpy as np
import pandas as pd

from src.forecast import spaces
from src.forecast.registry import BACKEND, SOURCE, family
from src.forecast.result import ForecastError, ForecastResult

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS where a GPU is used
for _name in ("pytorch_lightning", "lightning", "lightning.pytorch", "lightning_fabric"):
    logging.getLogger(_name).setLevel(logging.ERROR)


def backend_version(model: str) -> str:
    return metadata.version(BACKEND[family(model)])


def accelerator() -> str:
    """Device for neural fits: RERUN_ACCELERATOR = auto (default) | cpu | gpu."""
    choice = os.environ.get("RERUN_ACCELERATOR", "auto")
    if choice == "cpu":
        return "cpu"
    import torch

    has_gpu = torch.cuda.is_available()
    if choice == "gpu" and not has_gpu:
        raise ForecastError("RERUN_ACCELERATOR=gpu but no CUDA device is available")
    if choice not in ("auto", "gpu"):
        raise ValueError(f"RERUN_ACCELERATOR must be auto, cpu or gpu, not {choice!r}")
    return "gpu" if has_gpu else "cpu"


# --- statistical (local) ------------------------------------------------------------------

def statistical_model(model: str, season_length: int):
    from statsforecast.models import AutoARIMA, AutoETS

    if model == "ARIMA":
        return AutoARIMA(season_length=season_length, approximation=True)
    if model == "SARIMA":
        return AutoARIMA(season_length=season_length, D=1, approximation=True)
    if model == "ETS":
        return AutoETS(season_length=season_length)
    raise KeyError(f"{model} is not a statistical model")


def forecast_statistical(model: str, y_train, h: int, season_length: int, timings: dict | None = None) -> ForecastResult:
    """One series. Arguments as in the valid May run (regression test I2); raises on failure.

    fit + predict gives the same forecasts as the earlier runs' ``forecast`` (tested) and
    lets the two times be recorded separately in ``timings``.
    """
    from statsforecast import StatsForecast

    y = np.asarray(y_train, dtype=float)
    df = pd.DataFrame({"unique_id": ["s"] * len(y), "ds": range(len(y)), "y": y})
    sf = StatsForecast(models=[statistical_model(model, season_length)], freq=1, n_jobs=1)
    clock = {} if timings is None else timings
    start = time.perf_counter()
    sf.fit(df=df)
    clock["fit_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    fcst = sf.predict(h=h)
    clock["predict_seconds"] = time.perf_counter() - start
    col = [c for c in fcst.columns if c not in {"unique_id", "ds"}]
    if len(col) != 1:
        raise ForecastError(f"unexpected statsforecast output columns {list(fcst.columns)}")
    return ForecastResult(fcst[col[0]].to_numpy(dtype=float), SOURCE["statistical"], "statsforecast",
                          backend_version(model), h)


# --- global ----------------------------------------------------------------------------------

def _per_series(preds: pd.DataFrame, column: str, ids: list[str], h: int, model: str, key: str | None,
                seed: int, version: str, device: str) -> dict[str, ForecastResult]:
    if "unique_id" not in preds.columns:
        preds = preds.reset_index()
    out = {}
    grouped = {uid: g.sort_values("ds")[column].to_numpy(dtype=float) for uid, g in preds.groupby("unique_id", sort=False)}
    missing = [uid for uid in ids if uid not in grouped]
    if missing:
        raise ForecastError(f"{model}: no forecasts for {len(missing)} series, e.g. {missing[:3]}")
    for uid in ids:
        out[uid] = ForecastResult(grouped[uid], SOURCE[family(model)], BACKEND[family(model)], version, h,
                                  config_key=key, seed=seed, device=device)
    return out


def fit_predict_global(model: str, params: dict, train: pd.DataFrame, h: int, season_length: int, seed: int,
                       *, trained_steps: int | None = None, config_key: str | None = None,
                       timings: dict | None = None) -> dict[str, ForecastResult]:
    """Fit one global model on ``train`` (unique_id, ds = position t, y) and forecast h steps per series.

    ``timings``, when given, receives fit_seconds and predict_seconds.
    """
    fam = family(model)
    ids = sorted(train["unique_id"].unique())
    df = train[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    version = backend_version(model)
    clock = {} if timings is None else timings
    if fam == "ml":
        from mlforecast import MLForecast

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlf = MLForecast(
                models={model: spaces.ml_estimator(model, params, seed)}, freq=1,
                lags=spaces.ml_lags(season_length),
                target_transforms=spaces.ml_target_transforms(params, season_length),
            )
            start = time.perf_counter()
            mlf.fit(df, static_features=[])
            clock["fit_seconds"] = time.perf_counter() - start
            start = time.perf_counter()
            preds = mlf.predict(h)
            clock["predict_seconds"] = time.perf_counter() - start
        return _per_series(preds, model, ids, h, model, config_key, seed, version, "cpu")
    if fam in ("neural", "transformer"):
        from neuralforecast import NeuralForecast

        if trained_steps is None:
            raise ForecastError(f"{model}: the frozen number of training steps is required")
        device = accelerator()
        kwargs = spaces.neural_kwargs(model, params, h, seed, max_steps=trained_steps, accelerator=device)
        nf = NeuralForecast(models=[spaces.neural_class(model)(**kwargs)], freq=1)
        start = time.perf_counter()
        nf.fit(df, val_size=0)
        clock["fit_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        preds = nf.predict()
        clock["predict_seconds"] = time.perf_counter() - start
        return _per_series(preds, model, ids, h, model, config_key, seed, version, device)
    raise KeyError(f"{model} is not a global model")
