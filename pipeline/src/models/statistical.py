from __future__ import annotations

import numpy as np

from src.metrics.rel_naive import seasonal_naive_forecast
from src.models.forecast_result import ForecastResult
from src.training.strict_mode import StrictModeViolation, is_strict_mode


def forecast_statistical(model_name: str, y_train, h: int, season_length: int, mode: str = "no_finetune") -> ForecastResult:
    strict = is_strict_mode()
    # finetune_by_individual_series uses exact (non-approximated) fitting for AutoARIMA/AutoETS.
    approximation = mode != "finetune_by_individual_series"

    try:
        import statsforecast
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
        import pandas as pd

        backend_version = statsforecast.__version__
    except ImportError as exc:
        if strict:
            raise StrictModeViolation(
                f"statsforecast not available for {model_name!r}: {exc}"
            ) from exc
        base = seasonal_naive_forecast(y_train, h, season_length)
        return ForecastResult(
            yhat=base,
            model_output_source="deterministic_baseline_fallback",
            fit_status="fallback",
            fallback_reason=f"statsforecast_import_error:{exc.__class__.__name__}",
            model_backend="statsforecast",
            backend_library_version="unavailable",
        )

    autosarima_note: str | None = None
    if model_name == "AutoSARIMA":
        # AutoSARIMA: forces seasonal differencing (D=1), distinct from AutoARIMA which searches D.
        model_obj = AutoARIMA(season_length=season_length, D=1, approximation=approximation)
    else:
        model_map = {
            "AutoARIMA": AutoARIMA(season_length=season_length, approximation=approximation),
            "AutoETS": AutoETS(season_length=season_length),
        }
        model_obj = model_map.get(model_name, SeasonalNaive(season_length=season_length))

    try:
        y = list(map(float, y_train))
        df = pd.DataFrame({"unique_id": ["s"] * len(y), "ds": range(len(y)), "y": y})
        sf = StatsForecast(models=[model_obj], freq=1, n_jobs=1)
        fcst = sf.forecast(df=df, h=h)
        col = [c for c in fcst.columns if c not in {"unique_id", "ds"}][0]
        yhat = fcst[col].to_numpy(dtype=float)
        return ForecastResult(
            yhat=yhat,
            model_output_source="trained_statsforecast",
            fit_status="trained",
            fallback_reason=autosarima_note,
            model_backend="statsforecast",
            backend_library_version=backend_version,
        )
    except Exception as exc:
        # Known library limits that are data-driven, not code bugs — always fall back
        # regardless of strict mode so these never count as experiment failures.
        _known_limit = (
            isinstance(exc, NotImplementedError) and "tiny datasets" in str(exc)
        ) or (
            isinstance(exc, IndexError) and "size 0" in str(exc)
        )
        if strict and not _known_limit:
            raise StrictModeViolation(
                f"StatsForecast fit failed for {model_name!r}: {exc}"
            ) from exc
        base = seasonal_naive_forecast(y_train, h, season_length)
        return ForecastResult(
            yhat=base,
            model_output_source="deterministic_baseline_fallback",
            fit_status="fallback",
            fallback_reason=f"statsforecast_fit_failed:{exc.__class__.__name__}",
            model_backend="statsforecast",
            backend_library_version=backend_version,
        )
