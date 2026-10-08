from __future__ import annotations

import numpy as np

from .errors import DecompositionError, FeatureUndefined


def finite_array(y) -> np.ndarray:
    """The series as a float array. The frozen data is finite, so a non-finite value is an error."""
    arr = np.asarray(y, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("series contains non-finite values")
    return arr


def robust_scale(x: np.ndarray, eps: float = 1e-8) -> float:
    x = finite_array(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad + eps)


def decompose_series(y, season_length: int, periodic: bool = False, seasonal_deg: int = 1) -> dict[str, np.ndarray]:
    """Robust STL (protocol D9). No fallback: raises DecompositionError when STL cannot be fitted.

    The run's STL is statsmodels' default: a seasonal smoother of span 7 and degree 1.
    ``periodic`` (sensitivity analysis S1 only, change log v2.2): as R's ``s.window = "periodic"``,
    a seasonal smoother of length 10n + 1 and degree 0 on n points, then the seasonal
    component replaced by its mean at each position of the cycle (the remainder takes the
    difference), so the seasonal pattern is the same in every cycle.
    ``seasonal_deg=0`` (sensitivity analysis S2 only, change log v2.4): the run's STL with a
    seasonal smoother of degree 0 instead of 1, the span unchanged."""
    y = finite_array(y)
    if len(y) < 2 * season_length:
        raise DecompositionError("stl_too_short")
    if np.std(y) < 1e-10:
        raise DecompositionError("stl_constant_input")
    from statsmodels.tsa.seasonal import STL

    if periodic:
        res = STL(y, period=season_length, seasonal=10 * len(y) + 1, seasonal_deg=0, robust=True).fit()
        trend = np.asarray(res.trend)
        position = np.arange(len(y)) % season_length
        means = np.bincount(position, weights=np.asarray(res.seasonal)) / np.bincount(position)
        seasonal = means[position]
        return {"trend": trend, "seasonal": seasonal, "residual": y - trend - seasonal}
    elif seasonal_deg != 1:
        res = STL(y, period=season_length, seasonal_deg=seasonal_deg, robust=True).fit()
    else:
        res = STL(y, period=season_length, robust=True).fit()
    return {"trend": np.asarray(res.trend), "seasonal": np.asarray(res.seasonal), "residual": np.asarray(res.resid)}


def seasonal_strength(seasonal: np.ndarray, residual: np.ndarray) -> float:
    denom = np.var(seasonal + residual)
    if denom <= 1e-12:
        raise FeatureUndefined("seasonal_strength_undefined")
    return float(np.clip(1 - np.var(residual) / denom, 0, 1))


def non_normality_feature(y, season_length: int) -> float:
    y = finite_array(y)
    if len(y) < 20:
        raise FeatureUndefined("too_short")
    if np.std(y) < 1e-10:
        raise FeatureUndefined("constant")
    resid = decompose_series(y, season_length)["residual"]
    z = (resid - np.median(resid)) / robust_scale(resid)
    from scipy.stats import normaltest

    _, p = normaltest(z)
    return float(np.clip(-np.log10(float(p) + 1e-12), 0, 12))


def evolving_seasonality_feature(y, season_length: int) -> tuple[float, dict[str, float]]:
    y = finite_array(y)
    window = max(36, 3 * season_length) if season_length == 12 else max(16, 4 * season_length)
    step = max(1, season_length // 2)
    if len(y) < window + step:
        raise FeatureUndefined("insufficient_rolling_windows")
    strengths = []
    for start in range(0, len(y) - window + 1, step):
        dec = decompose_series(y[start : start + window], season_length)
        strengths.append(seasonal_strength(dec["seasonal"], dec["residual"]))
    x = np.arange(len(strengths))
    slope = float(np.polyfit(x, strengths, deg=1)[0])
    return float(np.std(strengths)), {
        "seasonal_strength_first": float(strengths[0]),
        "seasonal_strength_last": float(strengths[-1]),
        "seasonal_strength_slope": slope,
    }
