from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.features.errors import DecompositionError
from src.features.mstl_features import decompose_series


@dataclass
class STLTransform:
    season_length: int
    fitted_: bool = False
    n_train_: int = 0
    trend_: np.ndarray | None = None
    seasonal_: np.ndarray | None = None
    residual_: np.ndarray | None = None

    def fit(self, y_train) -> "STLTransform":
        dec = decompose_series(y_train, self.season_length)
        self.trend_ = dec["trend"]
        self.seasonal_ = dec["seasonal"]
        self.residual_ = dec["residual"]
        self.n_train_ = len(self.trend_)
        self.fitted_ = True
        return self

    def components(self) -> dict[str, np.ndarray]:
        if not self.fitted_:
            raise RuntimeError("STLTransform must be fitted before components are requested")
        return {"trend": self.trend_.copy(), "seasonal": self.seasonal_.copy(), "residual": self.residual_.copy()}

    def nonseasonal(self) -> np.ndarray:
        return self.trend_ + self.residual_

    def forecast_seasonal(self, h: int) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("STLTransform must be fitted before forecasting")
        seasonal = np.asarray(self.seasonal_, dtype=float)
        if len(seasonal) < self.season_length:
            # Unreachable after a successful fit (STL needs two seasons); never substitute values.
            raise DecompositionError("stl_too_short")
        last_cycle = seasonal[-self.season_length:]
        return np.asarray([last_cycle[i % self.season_length] for i in range(h)])

