"""What a model is trained on under each strategy (protocol Section 4.1, D9).

``component_targets`` decomposes one training history with robust STL (no fallback:
DecompositionError propagates) and returns every target: the raw series, T + R
(STL-SN), and T, S, R (STL-AC). It sees only the values it is given; callers pass a
window's training data in evaluation and a series' pre-window-0 history in tuning.

The STL variants of the sensitivity analyses apply to STL-SN only. S1 (change log v2.2):
``log`` decomposes log(y) (the targets are then on the log scale and the STL-SN forecast
is exp of the recomposition), ``periodic`` fixes the seasonal pattern over the window.
S2 (change log v2.4): ``deg0`` uses a seasonal smoother of degree 0; ``last3`` keeps the
run's decomposition and continues the mean of the last three seasonal cycles; ``stlf``
keeps the run's decomposition and continuation (the statistical model is fitted without
a seasonal component, see engine). The default is the run's STL (D9); its code path is
unchanged.
"""
from __future__ import annotations

import numpy as np

from src.features.mstl_features import DecompositionError, decompose_series

STL_VARIANTS = ("default", "log", "periodic", "deg0", "last3", "stlf")
RUN_DECOMPOSITION = ("default", "last3", "stlf")      # variants that use the run's STL unchanged
CONTINUATION_CYCLES = {"last3": 3}                    # seasonal cycles averaged by the continuation


def component_targets(y, season_length: int, variant: str = "default") -> dict[str, np.ndarray]:
    y = np.asarray(y, dtype=float)
    if variant in RUN_DECOMPOSITION:
        dec = decompose_series(y, season_length)
    elif variant == "log":
        if not (y > 0).all():
            raise DecompositionError("log_stl_nonpositive_input")
        dec = decompose_series(np.log(y), season_length)
    elif variant == "periodic":
        dec = decompose_series(y, season_length, periodic=True)
    elif variant == "deg0":
        dec = decompose_series(y, season_length, seasonal_deg=0)
    else:
        raise ValueError(f"unknown STL variant {variant!r}")
    return {
        "raw": y,
        "nonseasonal": dec["trend"] + dec["residual"],
        "trend": dec["trend"],
        "seasonal": dec["seasonal"],
        "residual": dec["residual"],
    }


def seasonal_continuation(seasonal: np.ndarray, h: int, season_length: int, cycles: int = 1) -> np.ndarray:
    """STL-SN: the last seasonal cycle of the training window, repeated over the horizon; with
    ``cycles`` > 1 (S2's ``last3``), the mean of the last ``cycles`` cycles, position by position."""
    seasonal = np.asarray(seasonal, dtype=float)
    if cycles == 1:
        last = seasonal[-season_length:]
        if len(last) != season_length:
            raise ValueError("the seasonal component is shorter than one season")
    else:
        if len(seasonal) < cycles * season_length:
            raise ValueError(f"the seasonal component is shorter than {cycles} seasons")
        last = seasonal[-cycles * season_length:].reshape(cycles, season_length).mean(axis=0)
    return np.asarray([last[i % season_length] for i in range(h)])


def stl_sn_forecast(nonseasonal_forecast, seasonal: np.ndarray, h: int, season_length: int,
                    variant: str = "default") -> np.ndarray:
    """STL-SN recomposition: the non-seasonal forecast plus the seasonal continuation, back
    on the original scale (exp) for the log variant."""
    cycles = CONTINUATION_CYCLES.get(variant, 1)
    yhat = nonseasonal_forecast + seasonal_continuation(seasonal, h, season_length, cycles)
    return np.exp(yhat) if variant == "log" else yhat
