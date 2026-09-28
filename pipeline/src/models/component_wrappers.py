from __future__ import annotations

import numpy as np

from src.models.baselines import model_forecast
from src.models.forecast_result import ForecastResult
from src.training.strict_mode import StrictModeViolation, is_strict_mode


def forecast_component(model_name: str, values, h: int, season_length: int) -> ForecastResult:
    if is_strict_mode():
        raise StrictModeViolation(
            f"Component forecast backend not implemented for {model_name!r}. "
            "Real component-level training required in strict mode."
        )
    return ForecastResult(
        yhat=np.asarray(model_forecast(model_name, values, h, season_length), dtype=float),
        model_output_source="deterministic_baseline_fallback",
        fit_status="fallback",
        fallback_reason="component_backend_not_implemented",
        model_backend="component_wrapper",
        backend_library_version="unavailable",
    )
