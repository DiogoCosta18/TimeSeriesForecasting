"""What a model is trained on under each strategy (protocol Section 4.1, D9).

``component_targets`` decomposes one training history with robust STL (no fallback:
DecompositionError propagates) and returns every target: the raw series, T + R
(STL-SN), and T, S, R (STL-AC). It sees only the values it is given; callers pass a
window's training data in evaluation and a series' pre-window-0 history in tuning.

The STL variants of the sensitivity analyses apply to STL-SN only, except ``p1``. S1 (change
log v2.2): ``log`` decomposes log(y) (the targets are then on the log scale and the STL-SN
forecast is exp of the recomposition), ``periodic`` fixes the seasonal pattern over the window.
S2 (change log v2.4): ``deg0`` uses a seasonal smoother of degree 0; ``last3`` keeps the
run's decomposition and continues the mean of the last three seasonal cycles; ``stlf``
keeps the run's decomposition and continuation (the statistical model is fitted without
a seasonal component, see engine). S3 and S4 (change log v2.12): ``s25`` uses a seasonal
smoother of span 25; a variant ending in ``p1`` (``p1``, ``periodic_p1``, ``s25_p1``) fits the
statistical models with season length 1 on the targets without seasonality, under STL-SN and,
for ``p1``, STL-AC, the decomposition being the variant's stem (``p1``: the run's STL). The
default is the run's STL (D9); its code path is unchanged.
"""
from __future__ import annotations

import numpy as np

from src.features.mstl_features import DecompositionError, decompose_series

DECOMPOSITIONS = ("default", "log", "periodic", "deg0", "last3", "s25")
STL_VARIANTS = (*DECOMPOSITIONS, "stlf", "p1", "periodic_p1", "s25_p1")
RUN_DECOMPOSITION = ("default", "last3")              # decompositions that use the run's STL unchanged
CONTINUATION_CYCLES = {"last3": 3}                    # seasonal cycles averaged by the continuation
WIDE_SEASONAL_SPAN = {"s25": 25}                      # seasonal smoother span of the wide variant


def decomposition_of(variant: str) -> str:
    """The STL a variant decomposes with: ``stlf`` and ``p1`` use the run's STL, ``X_p1`` that of ``X``."""
    if variant not in STL_VARIANTS:
        raise ValueError(f"unknown STL variant {variant!r}")
    if variant in ("stlf", "p1"):
        return "default"
    return variant[: -len("_p1")] if variant.endswith("_p1") else variant


def season_length_one(variant: str) -> bool:
    """Whether a variant fits the statistical models with season length 1 on the non-seasonal targets."""
    return variant in ("stlf", "p1") or variant.endswith("_p1")


def component_targets(y, season_length: int, variant: str = "default") -> dict[str, np.ndarray]:
    y = np.asarray(y, dtype=float)
    decomposition = decomposition_of(variant)
    if decomposition in RUN_DECOMPOSITION:
        dec = decompose_series(y, season_length)
    elif decomposition == "log":
        if not (y > 0).all():
            raise DecompositionError("log_stl_nonpositive_input")
        dec = decompose_series(np.log(y), season_length)
    elif decomposition == "periodic":
        dec = decompose_series(y, season_length, periodic=True)
    elif decomposition == "deg0":
        dec = decompose_series(y, season_length, seasonal_deg=0)
    elif decomposition in WIDE_SEASONAL_SPAN:
        dec = decompose_series(y, season_length, seasonal=WIDE_SEASONAL_SPAN[decomposition])
    else:  # pragma: no cover - decomposition_of refuses unknown variants
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
    decomposition = decomposition_of(variant)
    cycles = CONTINUATION_CYCLES.get(decomposition, 1)
    yhat = nonseasonal_forecast + seasonal_continuation(seasonal, h, season_length, cycles)
    return np.exp(yhat) if decomposition == "log" else yhat
