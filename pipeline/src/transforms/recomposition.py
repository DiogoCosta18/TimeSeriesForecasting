from __future__ import annotations

import numpy as np


def recompose_components(trend, seasonal, residual) -> np.ndarray:
    return np.asarray(trend, dtype=float) + np.asarray(seasonal, dtype=float) + np.asarray(residual, dtype=float)


def recompose_nonseasonal(nonseasonal_forecast, seasonal_forecast) -> np.ndarray:
    return np.asarray(nonseasonal_forecast, dtype=float) + np.asarray(seasonal_forecast, dtype=float)

