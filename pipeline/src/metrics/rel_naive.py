from __future__ import annotations

import numpy as np

from .losses import mae


def seasonal_naive_forecast(y_train, h: int, season_length: int) -> np.ndarray:
    """The last observed season, repeated; fewer than one season of history is refused (an
    eligible series always has more, D3), never replaced by another forecast."""
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) < season_length:
        raise ValueError(f"seasonal naive needs {season_length} training values, got {len(y_train)}")
    return np.asarray([y_train[-season_length + (i % season_length)] for i in range(h)], dtype=float)


def rel_naive(y_true, y_pred, y_naive, eps: float = 1e-8, clip: float | None = None) -> float:
    val = mae(y_true, y_pred) / (mae(y_true, y_naive) + eps)
    if clip is not None:
        val = min(val, clip)
    return float(val)

