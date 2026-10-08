"""What a model is trained on under each strategy (protocol Section 4.1, D9).

``component_targets`` decomposes one training history with robust STL (no fallback:
DecompositionError propagates) and returns every target: the raw series, T + R
(STL-SN), and T, S, R (STL-AC). It sees only the values it is given; callers pass a
window's training data in evaluation and a series' pre-window-0 history in tuning.

The STL variants of sensitivity analysis S1 (change log v2.2) apply to STL-SN only:
``log`` decomposes log(y) (the targets are then on the log scale and the STL-SN forecast
is exp of the recomposition), ``periodic`` fixes the seasonal pattern over the window.
The default is the run's STL (D9); its code path is unchanged.
"""
from __future__ import annotations

import numpy as np

from src.features.mstl_features import DecompositionError, decompose_series

STL_VARIANTS = ("default", "log", "periodic")


def component_targets(y, season_length: int, variant: str = "default") -> dict[str, np.ndarray]:
    y = np.asarray(y, dtype=float)
    if variant == "default":
        dec = decompose_series(y, season_length)
    elif variant == "log":
        if not (y > 0).all():
            raise DecompositionError("log_stl_nonpositive_input")
        dec = decompose_series(np.log(y), season_length)
    elif variant == "periodic":
        dec = decompose_series(y, season_length, periodic=True)
    else:
        raise ValueError(f"unknown STL variant {variant!r}")
    return {
        "raw": y,
        "nonseasonal": dec["trend"] + dec["residual"],
        "trend": dec["trend"],
        "seasonal": dec["seasonal"],
        "residual": dec["residual"],
    }


def seasonal_continuation(seasonal: np.ndarray, h: int, season_length: int) -> np.ndarray:
    """STL-SN: the last seasonal cycle of the training window, repeated over the horizon."""
    seasonal = np.asarray(seasonal, dtype=float)
    last = seasonal[-season_length:]
    if len(last) != season_length:
        raise ValueError("the seasonal component is shorter than one season")
    return np.asarray([last[i % season_length] for i in range(h)])


def stl_sn_forecast(nonseasonal_forecast, seasonal: np.ndarray, h: int, season_length: int,
                    variant: str = "default") -> np.ndarray:
    """STL-SN recomposition: the non-seasonal forecast plus the seasonal continuation, back
    on the original scale (exp) for the log variant."""
    yhat = nonseasonal_forecast + seasonal_continuation(seasonal, h, season_length)
    return np.exp(yhat) if variant == "log" else yhat
