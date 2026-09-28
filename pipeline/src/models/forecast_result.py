from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

ALLOWED_OUTPUT_SOURCES = frozenset(
    {
        "trained_statsforecast",
        "trained_mlforecast",
        "trained_neuralforecast",
        "trained_transformer",
        "deterministic_baseline_fallback",
        "placeholder_optuna",
        "failed_fallback",
        "strict_mode_failure",
        "skipped",
    }
)


@dataclass
class ForecastResult:
    yhat: np.ndarray
    model_output_source: str
    fit_status: str  # "trained" | "fallback" | "failed" | "skipped"
    fallback_reason: str | None = None
    model_backend: str | None = None
    backend_library_version: str | None = None
    device: str | None = None
    forecast_hash: str | None = field(default=None)

    def __post_init__(self) -> None:
        self.yhat = np.asarray(self.yhat, dtype=float)
        if self.forecast_hash is None:
            self.forecast_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        return hashlib.sha256(self.yhat.tobytes()).hexdigest()[:16]

    def with_adjusted_yhat(self, adjusted: np.ndarray) -> "ForecastResult":
        return ForecastResult(
            yhat=adjusted,
            model_output_source=self.model_output_source,
            fit_status=self.fit_status,
            fallback_reason=self.fallback_reason,
            model_backend=self.model_backend,
            backend_library_version=self.backend_library_version,
            device=self.device,
            # forecast_hash is recomputed automatically via __post_init__
        )
