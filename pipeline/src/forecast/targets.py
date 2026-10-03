"""What a model is trained on under each strategy (protocol Section 4.1, D9).

``component_targets`` decomposes one training history with robust STL (no fallback:
DecompositionError propagates) and returns every target: the raw series, T + R
(STL-SN), and T, S, R (STL-AC). It sees only the values it is given; callers pass a
window's training data in evaluation and a series' pre-window-0 history in tuning.
"""
from __future__ import annotations

import numpy as np

from src.features.mstl_features import decompose_series


def component_targets(y, season_length: int) -> dict[str, np.ndarray]:
    y = np.asarray(y, dtype=float)
    dec = decompose_series(y, season_length)
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
