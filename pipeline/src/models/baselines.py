from __future__ import annotations

import numpy as np

from src.metrics.rel_naive import seasonal_naive_forecast


def mean_forecast(y_train, h: int) -> np.ndarray:
    y = np.asarray(y_train, dtype=float)
    return np.repeat(float(np.nanmean(y)) if len(y) else 0.0, h)


def drift_forecast(y_train, h: int) -> np.ndarray:
    y = np.asarray(y_train, dtype=float)
    if len(y) < 2:
        return mean_forecast(y, h)
    slope = (y[-1] - y[0]) / max(len(y) - 1, 1)
    return y[-1] + slope * np.arange(1, h + 1)


def damped_trend_forecast(y_train, h: int, damping: float = 0.85) -> np.ndarray:
    y = np.asarray(y_train, dtype=float)
    if len(y) < 3:
        return drift_forecast(y, h)
    slope = np.median(np.diff(y[-min(len(y), 12) :]))
    steps = np.array([(1 - damping ** i) / (1 - damping) for i in range(1, h + 1)])
    return y[-1] + slope * steps


def model_forecast(model_name: str, y_train, h: int, season_length: int) -> np.ndarray:
    name = model_name.lower()
    if "ets" in name or "nhits" in name or "tft" in name:
        return 0.65 * seasonal_naive_forecast(y_train, h, season_length) + 0.35 * damped_trend_forecast(y_train, h)
    if "arima" in name or "linear" in name or "ridge" in name or "nlinear" in name or "patch" in name:
        return 0.5 * seasonal_naive_forecast(y_train, h, season_length) + 0.5 * drift_forecast(y_train, h)
    if "forest" in name or "xgboost" in name or "lstm" in name:
        return 0.75 * seasonal_naive_forecast(y_train, h, season_length) + 0.25 * mean_forecast(y_train, h)
    return seasonal_naive_forecast(y_train, h, season_length)

