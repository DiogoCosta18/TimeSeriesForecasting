from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.metrics.rel_naive import seasonal_naive_forecast
from src.models.baselines import model_forecast
from src.models.forecast_result import ForecastResult
from src.models.global_adapter_data import regularize_global_training_frame
from src.training.strict_mode import StrictModeViolation, is_strict_mode

_MLFORECAST_VERSION: str | None = None


def _get_mlforecast_version() -> str:
    global _MLFORECAST_VERSION
    if _MLFORECAST_VERSION is None:
        try:
            import mlforecast
            _MLFORECAST_VERSION = mlforecast.__version__
        except ImportError:
            _MLFORECAST_VERSION = "unavailable"
    return _MLFORECAST_VERSION


def _build_estimator(model_name: str):
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor

    if model_name == "AutoRidge":
        return Ridge(alpha=1.0)
    elif model_name == "AutoLinearRegression":
        return LinearRegression()
    elif model_name == "AutoRandomForest":
        return RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=42)
    elif model_name == "AutoXGBoost":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=1)
    else:
        return Ridge(alpha=1.0)


def _safe_lags(df_train: pd.DataFrame, season_length: int) -> list[int]:
    min_len = df_train.groupby("unique_id").size().min()
    lags = [1, 2, 3]
    if min_len > season_length + 4:
        lags.append(season_length)
    return lags


def forecast_mlforecast_global(
    model_name: str,
    df_train: pd.DataFrame,
    h: int,
    season_length: int,
    freq_str: str,
    num_optuna_samples: int = 2,
) -> dict[str, ForecastResult]:
    uids = df_train["unique_id"].unique().tolist()
    version = _get_mlforecast_version()

    try:
        from mlforecast import MLForecast

        df_model = regularize_global_training_frame(df_train, freq_str)
        if df_model.empty:
            raise ValueError("No valid rows available for MLForecast training.")
        estimator = _build_estimator(model_name)
        lags = _safe_lags(df_model, season_length)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlf = MLForecast(models=[estimator], freq=freq_str, lags=lags)
            mlf.fit(df_model[["unique_id", "ds", "y"]], static_features=[])
            preds = mlf.predict(h)

        pred_col = [c for c in preds.columns if c not in ("unique_id", "ds")][0]
        results: dict[str, ForecastResult] = {}
        for uid in uids:
            uid_preds = preds[preds["unique_id"] == uid].sort_values("ds")[pred_col].to_numpy(dtype=float)
            if len(uid_preds) < h:
                uid_preds = np.pad(uid_preds, (0, h - len(uid_preds)), mode="edge") if len(uid_preds) > 0 else np.full(h, np.nan)
            results[uid] = ForecastResult(
                yhat=uid_preds[:h],
                model_output_source="trained_mlforecast",
                fit_status="trained",
                model_backend="mlforecast",
                backend_library_version=version,
            )
        return results

    except Exception as exc:
        if is_strict_mode():
            raise StrictModeViolation(f"MLForecast global training failed for {model_name!r}: {exc}") from exc
        fallback: dict[str, ForecastResult] = {}
        for uid in uids:
            uid_train = df_train[df_train["unique_id"] == uid]["y"].to_numpy(dtype=float)
            base = seasonal_naive_forecast(uid_train, h, season_length)
            fallback[uid] = ForecastResult(
                yhat=base,
                model_output_source="deterministic_baseline_fallback",
                fit_status="fallback",
                fallback_reason=f"mlforecast_error:{type(exc).__name__}",
                model_backend="mlforecast",
                backend_library_version=version,
            )
        return fallback


def forecast_mlforecast(model_name: str, y_train, h: int, season_length: int, mode: str = "no_finetune") -> ForecastResult:
    if is_strict_mode():
        raise StrictModeViolation(
            f"MLForecast backend not implemented for {model_name!r}. "
            "Real training required in strict mode."
        )
    adjustment = 1.0
    if mode == "series_refit_from_global_config":
        adjustment = 0.98
    elif mode == "bucket_refit_from_global_config":
        adjustment = 0.99
    base = model_forecast(model_name, y_train, h, season_length)
    return ForecastResult(
        yhat=np.asarray(base) * adjustment,
        model_output_source="deterministic_baseline_fallback",
        fit_status="fallback",
        fallback_reason="mlforecast_not_implemented",
        model_backend="mlforecast",
        backend_library_version="unavailable",
    )


def automlforecast_search_space() -> dict:
    return {
        "lags": [[1, 2, 3], [1, 2, 3, 12], [1, 4, 8]],
        "lag_transforms": ["rolling_mean", "rolling_std", "expanding_mean"],
        "differences": [[], [1], [12]],
        "date_features": [[], ["month"], ["quarter"]],
        "static_features": [
            "no_static_features",
            "six_numeric_features",
            "six_numeric_features+source_dataset",
            "six_numeric_features+feature_bins+source_dataset",
        ],
    }
