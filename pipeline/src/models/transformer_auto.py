from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics.rel_naive import seasonal_naive_forecast
from src.models.baselines import model_forecast
from src.models.forecast_result import ForecastResult
from src.models.global_adapter_data import regularize_global_training_frame
from src.models.torch_device import lightning_trainer_device_kwargs
from src.training.strict_mode import StrictModeViolation, is_strict_mode

_NEURALFORECAST_VERSION: str | None = None


def _get_neuralforecast_version() -> str:
    global _NEURALFORECAST_VERSION
    if _NEURALFORECAST_VERSION is None:
        try:
            import neuralforecast
            _NEURALFORECAST_VERSION = neuralforecast.__version__
        except ImportError:
            _NEURALFORECAST_VERSION = "unavailable"
    return _NEURALFORECAST_VERSION


def _safe_input_size(h: int, season_length: int, min_series_length: int) -> int:
    desired = max(h, min(2 * h, season_length))
    return max(2, min(desired, max(2, min_series_length - 1)))


def _build_transformer_model(model_name: str, h: int, season_length: int, num_optuna_samples: int, min_series_length: int):
    device_kwargs = lightning_trainer_device_kwargs()
    from neuralforecast.models import TFT, PatchTST

    input_size = _safe_input_size(h, season_length, min_series_length)
    max_steps = max(10, num_optuna_samples * 5)
    common = {
        "h": h,
        "input_size": input_size,
        "max_steps": max_steps,
        "batch_size": 16,
        "windows_batch_size": 64,
        "start_padding_enabled": True,
        "enable_progress_bar": False,
        "logger": False,
        **device_kwargs,
    }

    if model_name == "AutoPatchTST":
        patch_len = max(2, min(16, input_size))
        return PatchTST(
            **common,
            hidden_size=32,
            n_heads=4,
            encoder_layers=1,
            linear_hidden_size=64,
            patch_len=patch_len,
            stride=max(1, patch_len // 2),
        )
    else:
        return TFT(**common, hidden_size=32, n_head=4)


def forecast_transformer_global(
    model_name: str,
    df_train: pd.DataFrame,
    h: int,
    season_length: int,
    freq_str: str,
    num_optuna_samples: int = 2,
) -> dict[str, ForecastResult]:
    uids = df_train["unique_id"].unique().tolist()
    version = _get_neuralforecast_version()

    try:
        from neuralforecast import NeuralForecast

        df_model = regularize_global_training_frame(df_train, freq_str)
        if df_model.empty:
            raise ValueError("No valid rows available for Transformer training.")
        min_series_length = int(df_model.groupby("unique_id").size().min())
        transformer_model = _build_transformer_model(model_name, h, season_length, num_optuna_samples, min_series_length)
        nf = NeuralForecast(models=[transformer_model], freq=freq_str)
        nf.fit(df_model[["unique_id", "ds", "y"]], val_size=0)
        preds = nf.predict()

        if isinstance(preds.index, pd.MultiIndex) or preds.index.name == "unique_id":
            preds = preds.reset_index()
        if "unique_id" not in preds.columns:
            preds = preds.reset_index()

        pred_col = [c for c in preds.columns if c not in ("unique_id", "ds")][0]

        results: dict[str, ForecastResult] = {}
        for uid in uids:
            uid_preds = preds[preds["unique_id"] == uid].sort_values("ds")[pred_col].to_numpy(dtype=float)
            if len(uid_preds) < h:
                uid_preds = np.pad(uid_preds, (0, h - len(uid_preds)), mode="edge") if len(uid_preds) > 0 else np.full(h, np.nan)
            results[uid] = ForecastResult(
                yhat=uid_preds[:h],
                model_output_source="trained_transformer",
                fit_status="trained",
                model_backend="neuralforecast",
                backend_library_version=version,
            )
        return results

    except Exception as exc:
        if is_strict_mode():
            raise StrictModeViolation(f"Transformer global training failed for {model_name!r}: {exc}") from exc
        fallback: dict[str, ForecastResult] = {}
        for uid in uids:
            uid_train = df_train[df_train["unique_id"] == uid]["y"].to_numpy(dtype=float)
            base = seasonal_naive_forecast(uid_train, h, season_length)
            fallback[uid] = ForecastResult(
                yhat=base,
                model_output_source="deterministic_baseline_fallback",
                fit_status="fallback",
                fallback_reason=f"transformer_error:{type(exc).__name__}",
                model_backend="neuralforecast",
                backend_library_version=version,
            )
        return fallback


def forecast_transformer(model_name: str, y_train, h: int, season_length: int, mode: str = "no_finetune") -> ForecastResult:
    if is_strict_mode():
        raise StrictModeViolation(
            f"Transformer backend not implemented for {model_name!r}. "
            "Real training required in strict mode."
        )
    shrink = {"no_finetune": 1.0, "finetune_by_feature_bucket": 0.982, "finetune_by_series": 0.972}.get(mode, 1.0)
    return ForecastResult(
        yhat=np.asarray(model_forecast(model_name, y_train, h, season_length)) * shrink,
        model_output_source="deterministic_baseline_fallback",
        fit_status="fallback",
        fallback_reason="transformer_not_implemented",
        model_backend="transformer",
        backend_library_version="unavailable",
    )


def transformer_auto_classes() -> dict[str, str]:
    return {"AutoTFT": "AutoTFT", "AutoPatchTST": "AutoPatchTST"}
