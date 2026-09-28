from __future__ import annotations

import numpy as np


def clean_array(y) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.size == 0:
        return arr
    mask = np.isfinite(arr)
    if mask.all():
        return arr
    if not mask.any():
        return np.zeros_like(arr)
    idx = np.arange(arr.size)
    arr[~mask] = np.interp(idx[~mask], idx[mask], arr[mask])
    return arr


def robust_scale(x: np.ndarray, eps: float = 1e-8) -> float:
    x = clean_array(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad + eps)


def decompose_series(y, season_length: int) -> dict[str, np.ndarray]:
    y = clean_array(y)
    n = len(y)
    if n == 0:
        return {"trend": y, "seasonal": y, "residual": y}
    if n < max(2 * season_length, 8) or np.nanstd(y) < 1e-10:
        trend = np.full(n, float(np.nanmean(y) if n else 0.0))
        seasonal = np.zeros(n)
        residual = y - trend
        return {"trend": trend, "seasonal": seasonal, "residual": residual}
    try:
        from statsmodels.tsa.seasonal import STL

        res = STL(y, period=season_length, robust=True).fit()
        return {"trend": np.asarray(res.trend), "seasonal": np.asarray(res.seasonal), "residual": np.asarray(res.resid)}
    except Exception:
        kernel = max(3, season_length)
        trend = np.convolve(y, np.ones(kernel) / kernel, mode="same")
        seasonal = np.zeros(n)
        for s in range(season_length):
            idx = np.arange(s, n, season_length)
            seasonal[idx] = np.mean(y[idx] - trend[idx])
        residual = y - trend - seasonal
        return {"trend": trend, "seasonal": seasonal, "residual": residual}


def seasonal_strength(seasonal: np.ndarray, residual: np.ndarray) -> float:
    denom = np.var(seasonal + residual)
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(1 - np.var(residual) / denom, 0, 1))


def non_normality_feature(y, season_length: int) -> tuple[float, str]:
    y = clean_array(y)
    if len(y) < 20 or np.std(y) < 1e-10:
        return 0.0, "fallback_short_or_constant"
    resid = decompose_series(y, season_length)["residual"]
    z = (resid - np.median(resid)) / robust_scale(resid)
    try:
        from scipy.stats import normaltest

        if len(z) < 8:
            return 0.0, "fallback_short_or_constant"
        _, p = normaltest(z)
        return float(np.clip(-np.log10(float(p) + 1e-12), 0, 12)), "ok"
    except Exception:
        return float(np.clip(abs(np.mean(z**3)) + abs(np.mean(z**4) - 3), 0, 12)), "fallback_moments"


def evolving_seasonality_feature(y, season_length: int) -> tuple[float, str, dict[str, float]]:
    y = clean_array(y)
    window = max(36, 3 * season_length) if season_length == 12 else max(16, 4 * season_length)
    step = max(1, season_length // 2)
    if len(y) < window + step:
        return 0.0, "insufficient_rolling_windows", {}
    strengths = []
    for start in range(0, len(y) - window + 1, step):
        dec = decompose_series(y[start : start + window], season_length)
        strengths.append(seasonal_strength(dec["seasonal"], dec["residual"]))
    if len(strengths) < 2:
        return 0.0, "insufficient_rolling_windows", {}
    x = np.arange(len(strengths))
    slope = float(np.polyfit(x, strengths, deg=1)[0]) if len(strengths) > 1 else 0.0
    return float(np.std(strengths)), "ok", {
        "seasonal_strength_first": float(strengths[0]),
        "seasonal_strength_last": float(strengths[-1]),
        "seasonal_strength_slope": slope,
    }

