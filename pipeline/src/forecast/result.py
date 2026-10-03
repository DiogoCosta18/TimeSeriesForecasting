"""The only object that carries a forecast (protocol Section 4.5; test U9).

A ForecastResult holds forecasts that a trained model produced: its source must be one
of TRAINED_SOURCES, the values finite and exactly h long. A failure is not a result:
it is recorded as a failed row by the evaluation engine, without any forecast.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

TRAINED_SOURCES = frozenset({"trained_statsforecast", "trained_mlforecast", "trained_neuralforecast"})


class ForecastError(RuntimeError):
    """A model did not produce a valid forecast."""


def forecast_hash(values: np.ndarray) -> str:
    """SHA-256 of the forecasts as little-endian float64 (stored so reruns compare exactly)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes()).hexdigest()


@dataclass(frozen=True)
class ForecastResult:
    yhat: np.ndarray
    source: str
    backend: str
    backend_version: str
    h: int
    config_key: str | None = None
    seed: int | None = None
    device: str = "cpu"
    forecast_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.source not in TRAINED_SOURCES:
            raise ForecastError(f"forecast source {self.source!r} is not a trained model ({sorted(TRAINED_SOURCES)})")
        values = np.asarray(self.yhat, dtype=float)
        if values.shape != (self.h,):
            raise ForecastError(f"expected {self.h} forecasts, got shape {values.shape}")
        if not np.isfinite(values).all():
            raise ForecastError("non-finite forecasts")
        object.__setattr__(self, "yhat", values)
        object.__setattr__(self, "forecast_hash", forecast_hash(values))
