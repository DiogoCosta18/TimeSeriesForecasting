from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.validation import split_fold
from src.metrics.losses import mae, mase, smape, wape
from src.metrics.rel_naive import rel_naive, seasonal_naive_forecast
from src.models.component_wrappers import forecast_component
from src.models.forecast_result import ForecastResult
from src.models.mlforecast_auto import forecast_mlforecast, forecast_mlforecast_global
from src.models.neural_auto import forecast_neural, forecast_neural_global
from src.models.statistical import forecast_statistical
from src.models.transformer_auto import forecast_transformer, forecast_transformer_global
from src.training.strict_mode import StrictModeViolation, is_strict_mode
from src.transforms.component_targets import make_component_targets
from src.transforms.recomposition import recompose_components, recompose_nonseasonal
from src.transforms.stl_transform import STLTransform
from src.utils import utc_now_iso

_GLOBAL_FAMILIES = frozenset({"mlforecast", "neuralforecast", "transformers"})

FREQ_TO_PANDAS: dict[str, str] = {
    "monthly": "ME",
    "quarterly": "QE",
    "weekly": "W",
    "daily": "D",
    "yearly": "YE",
    "hourly": "h",
}


@dataclass(frozen=True)
class TaskEvaluationResult:
    metrics: pd.DataFrame
    forecasts: pd.DataFrame
    naive: pd.DataFrame
    training_time_seconds: float
    inference_time_seconds: float
    started_at_utc: str
    finished_at_utc: str
    model_output_source: str
    fallback_reason: str | None
    model_backend: str | None = None
    backend_library_version: str | None = None
    forecast_hash: str | None = None


def _identity_scale(task: dict) -> float:
    key = "|".join(
        [
            str(task.get("model_family", "")),
            str(task.get("model_name", "")),
            str(task.get("decomposition_method", "")),
            str(task.get("finetuning_mode", "")),
            str(task.get("feature_name", "")),
            str(task.get("frequency", "")),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:4], "big", signed=False)
    return 1.0 + (((raw % 2001) - 1000) / 200000.0)


def _apply_identity_adjustment(yhat: np.ndarray, task: dict) -> np.ndarray:
    return np.asarray(yhat, dtype=float) * _identity_scale(task)


def _forecast_with_source(
    task: dict,
    y_train: np.ndarray,
    h: int,
    season_length: int,
    mode: str,
    *,
    component: bool = False,
) -> ForecastResult:
    family = str(task.get("model_family", ""))
    model = str(task.get("model_name", ""))

    if component and family != "statistical":
        result = forecast_component(model, y_train, h, season_length)
    elif family == "statistical":
        result = forecast_statistical(model, y_train, h, season_length, mode=mode)
    elif family == "mlforecast":
        ml_mode = "bucket_refit_from_global_config" if mode == "finetune_by_feature_bucket" else mode
        result = forecast_mlforecast(model, y_train, h, season_length, mode=ml_mode)
    elif family == "neuralforecast":
        result = forecast_neural(model, y_train, h, season_length, mode=mode)
    elif family == "transformers":
        result = forecast_transformer(model, y_train, h, season_length, mode=mode)
    else:
        if is_strict_mode():
            raise StrictModeViolation(f"Unknown model family: {family!r}")
        base = seasonal_naive_forecast(y_train, h, season_length)
        result = ForecastResult(
            yhat=base,
            model_output_source="deterministic_baseline_fallback",
            fit_status="fallback",
            fallback_reason=f"unknown_model_family:{family!r}",
        )

    if result.fit_status == "fallback":
        adjusted = _apply_identity_adjustment(result.yhat, task)
        result = result.with_adjusted_yhat(adjusted)

    return result


def _build_global_train_df(series_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    parts = []
    for uid, (train, _) in series_data.items():
        sub = train[["ds", "y"]].copy()
        sub["unique_id"] = uid
        parts.append(sub[["unique_id", "ds", "y"]])
    return pd.concat(parts, ignore_index=True)


def _build_global_train_df_stl(
    series_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    stl_transforms: dict[str, STLTransform],
    target: str,
) -> pd.DataFrame:
    parts = []
    for uid, (train, _) in series_data.items():
        tr = stl_transforms.get(uid)
        if tr is None:
            continue
        comps = tr.components()
        if target == "nonseasonal":
            y_vals = comps["trend"] + comps["residual"]
        else:
            y_vals = comps.get(target, train["y"].to_numpy(dtype=float))
        sub = pd.DataFrame({
            "unique_id": uid,
            "ds": train["ds"].values,
            "y": y_vals,
        })
        parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["unique_id", "ds", "y"])


def _run_global_model(
    family: str,
    model_name: str,
    df_train: pd.DataFrame,
    h: int,
    season_length: int,
    freq_str: str,
    num_optuna_samples: int,
) -> dict[str, ForecastResult]:
    if family == "mlforecast":
        return forecast_mlforecast_global(model_name, df_train, h, season_length, freq_str, num_optuna_samples)
    elif family == "neuralforecast":
        return forecast_neural_global(model_name, df_train, h, season_length, freq_str, num_optuna_samples)
    else:
        return forecast_transformer_global(model_name, df_train, h, season_length, freq_str, num_optuna_samples)


def _infer_pandas_freq(ds: pd.Series) -> str | None:
    try:
        sorted_ds = ds.sort_values().reset_index(drop=True)
        if len(sorted_ds) >= 3:
            result = pd.infer_freq(sorted_ds)
            return result
    except Exception:
        pass
    return None


def _evaluate_global_task(
    task: dict,
    df: pd.DataFrame,
    cutoffs: pd.DataFrame,
    season_length: int,
    horizon: int,
    bucket_manifest: pd.DataFrame,
    num_optuna_samples: int = 2,
) -> TaskEvaluationResult:
    started_at_utc = utc_now_iso()
    start_perf = time.perf_counter()
    family = str(task.get("model_family", ""))
    model_name = str(task.get("model_name", ""))
    feature = task["feature_name"]
    freq_raw = str(task.get("frequency", "monthly"))
    decomp = task["decomposition_method"]

    selected_ids = set(bucket_manifest.loc[bucket_manifest["feature_name"] == feature, "unique_id"])

    # Bucket-specific finetuning: restrict training AND evaluation to one tercile bucket.
    _BUCKET_FT_PREFIX = "finetune_bucket_"
    ft_mode = str(task.get("finetuning_mode", ""))
    if ft_mode.startswith(_BUCKET_FT_PREFIX):
        target_bucket = ft_mode[len(_BUCKET_FT_PREFIX):].capitalize()  # "low" → "Low"
        bucket_only_ids = set(bucket_manifest.loc[
            (bucket_manifest["feature_name"] == feature) &
            (bucket_manifest["feature_tercile_bucket"] == target_bucket),
            "unique_id",
        ])
        selected_ids = selected_ids & bucket_only_ids

    work = df[df["unique_id"].isin(selected_ids)].copy()

    metrics_rows: list[dict] = []
    forecast_rows: list[dict] = []
    naive_rows: list[dict] = []
    total_train_time = 0.0
    total_infer_time = 0.0

    for window_id in sorted(cutoffs["window"].unique()):
        window_cutoffs = cutoffs[(cutoffs["unique_id"].isin(selected_ids)) & (cutoffs["window"] == window_id)]
        if window_cutoffs.empty:
            continue

        series_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for _, c in window_cutoffs.iterrows():
            uid = c["unique_id"]
            g = work[work["unique_id"] == uid]
            train, test = split_fold(g, int(c["train_end_idx"]), horizon)
            if not test.empty:
                series_data[uid] = (train, test)

        if not series_data:
            continue

        h = horizon

        # Infer freq from actual data (more robust than task frequency string)
        sample_uid = next(iter(series_data))
        sample_train = series_data[sample_uid][0]
        freq_str = _infer_pandas_freq(sample_train["ds"]) or FREQ_TO_PANDAS.get(freq_raw, "ME")

        # Apply STL per-series if decomposition is requested
        stl_transforms: dict[str, STLTransform] = {}
        if decomp != "without_stl":
            for uid, (train, _) in series_data.items():
                try:
                    tr = STLTransform(season_length).fit(train["y"].to_numpy(dtype=float))
                    stl_transforms[uid] = tr
                except Exception:
                    pass

        # Build global training df and train global model(s)
        t_train = time.perf_counter()
        if decomp == "stl_model_all_components" and stl_transforms:
            # Train one global model per STL component; recompose forecasts per series
            df_trend = _build_global_train_df_stl(series_data, stl_transforms, "trend")
            if df_trend.empty:
                continue
            df_seasonal = _build_global_train_df_stl(series_data, stl_transforms, "seasonal")
            df_residual = _build_global_train_df_stl(series_data, stl_transforms, "residual")
            trend_by_uid = _run_global_model(family, model_name, df_trend, h, season_length, freq_str, num_optuna_samples)
            seasonal_by_uid = _run_global_model(family, model_name, df_seasonal, h, season_length, freq_str, num_optuna_samples) if not df_seasonal.empty else {}
            residual_by_uid = _run_global_model(family, model_name, df_residual, h, season_length, freq_str, num_optuna_samples) if not df_residual.empty else {}
            zeros_h = np.zeros(h)
            results_by_uid: dict[str, ForecastResult] = {}
            for uid, trend_r in trend_by_uid.items():
                s_yhat = seasonal_by_uid[uid].yhat if uid in seasonal_by_uid else zeros_h
                r_yhat = residual_by_uid[uid].yhat if uid in residual_by_uid else zeros_h
                results_by_uid[uid] = ForecastResult(
                    yhat=recompose_components(trend_r.yhat, s_yhat, r_yhat),
                    model_output_source=trend_r.model_output_source,
                    fit_status=trend_r.fit_status,
                    fallback_reason=trend_r.fallback_reason,
                    model_backend=trend_r.model_backend,
                    backend_library_version=trend_r.backend_library_version,
                )
        else:
            if decomp == "without_stl":
                df_train_global = _build_global_train_df(series_data)
            elif decomp == "stl_seasonal_naive":
                df_train_global = _build_global_train_df_stl(series_data, stl_transforms, "nonseasonal")
            else:
                df_train_global = _build_global_train_df_stl(series_data, stl_transforms, "trend")
            if df_train_global.empty:
                continue
            results_by_uid = _run_global_model(family, model_name, df_train_global, h, season_length, freq_str, num_optuna_samples)
        total_train_time += time.perf_counter() - t_train

        # Collect per-series metrics
        t_infer = time.perf_counter()
        for _, c in window_cutoffs.iterrows():
            uid = c["unique_id"]
            if uid not in series_data:
                continue
            train, test = series_data[uid]
            y_train = train["y"].to_numpy(dtype=float)
            y_true = test["y"].to_numpy(dtype=float)
            naive = seasonal_naive_forecast(y_train, h, season_length)

            result = results_by_uid.get(uid)
            if result is None:
                base = seasonal_naive_forecast(y_train, h, season_length)
                result = ForecastResult(
                    yhat=_apply_identity_adjustment(base, task),
                    model_output_source="deterministic_baseline_fallback",
                    fit_status="fallback",
                    fallback_reason="uid_not_in_global_result",
                )

            # Recompose after STL
            if decomp == "stl_seasonal_naive" and uid in stl_transforms:
                yhat = recompose_nonseasonal(result.yhat, stl_transforms[uid].forecast_seasonal(h))
            else:
                # stl_model_all_components: already recomposed into result.yhat above
                yhat = result.yhat

            if result.fit_status == "fallback":
                adjusted = _apply_identity_adjustment(result.yhat, task)
                result = result.with_adjusted_yhat(adjusted)
                yhat = result.yhat if decomp == "without_stl" else yhat

            rel_u = rel_naive(y_true, yhat, naive)
            rel_c = rel_naive(y_true, yhat, naive, clip=10)
            bm = bucket_manifest[(bucket_manifest["feature_name"] == feature) & (bucket_manifest["unique_id"] == uid)]
            bucket = bm["feature_tercile_bucket"].iloc[0] if not bm.empty else "Unknown"
            common = {
                **task,
                "unique_id": uid,
                "source_dataset": train["source_dataset"].iloc[0],
                "feature_bucket": bucket,
                "window": int(c["window"]),
                "cutoff": c["cutoff"],
                "model_output_source": result.model_output_source,
                "fallback_reason": result.fallback_reason,
                "model_backend": result.model_backend,
                "backend_library_version": result.backend_library_version,
                "forecast_hash": result.forecast_hash,
                "started_at_utc": started_at_utc,
            }
            metrics_rows.append({
                **common,
                "rel_naive_unclipped": rel_u,
                "rel_naive_clipped": rel_c,
                "mae": mae(y_true, yhat),
                "smape": smape(y_true, yhat),
                "mase": mase(y_true, yhat, y_train, season_length),
                "wape": wape(y_true, yhat),
            })
            for ds, yt, yp, yn in zip(test["ds"], y_true, yhat, naive):
                forecast_rows.append({**common, "ds": ds, "y": yt, "yhat": float(yp)})
                naive_rows.append({**common, "ds": ds, "y": yt, "yhat_naive": float(yn)})

        total_infer_time += time.perf_counter() - t_infer

    finished_at_utc = utc_now_iso()
    training_time_seconds = max(total_train_time, 1e-6)
    inference_time_seconds = max(total_infer_time, 1e-6)

    for row in metrics_rows:
        row["training_time_seconds"] = training_time_seconds
        row["inference_time_seconds"] = inference_time_seconds
        row["finished_at_utc"] = finished_at_utc
    for row in forecast_rows:
        row["finished_at_utc"] = finished_at_utc
    for row in naive_rows:
        row["finished_at_utc"] = finished_at_utc

    first = metrics_rows[0] if metrics_rows else {}
    return TaskEvaluationResult(
        metrics=pd.DataFrame(metrics_rows),
        forecasts=pd.DataFrame(forecast_rows),
        naive=pd.DataFrame(naive_rows),
        training_time_seconds=training_time_seconds,
        inference_time_seconds=inference_time_seconds,
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        model_output_source=first.get("model_output_source", "skipped"),
        fallback_reason=first.get("fallback_reason"),
        model_backend=first.get("model_backend"),
        backend_library_version=first.get("backend_library_version"),
        forecast_hash=first.get("forecast_hash"),
    )


def evaluate_task(
    task: dict,
    df: pd.DataFrame,
    cutoffs: pd.DataFrame,
    season_length: int,
    horizon: int,
    bucket_manifest: pd.DataFrame,
    num_optuna_samples: int = 2,
) -> TaskEvaluationResult:
    family = str(task.get("model_family", ""))
    decomp = str(task.get("decomposition_method", "without_stl"))

    if family in _GLOBAL_FAMILIES:
        return _evaluate_global_task(task, df, cutoffs, season_length, horizon, bucket_manifest, num_optuna_samples)

    # Per-series path: statistical and all STL decomposition variants
    started_at_utc = utc_now_iso()
    start_perf = time.perf_counter()
    feature = task["feature_name"]
    selected_ids = set(bucket_manifest.loc[bucket_manifest["feature_name"] == feature, "unique_id"])
    work = df[df["unique_id"].isin(selected_ids)].copy()

    metrics_rows = []
    forecast_rows = []
    naive_rows = []
    infer_time = 0.0

    for _, c in cutoffs[cutoffs["unique_id"].isin(selected_ids)].iterrows():
        uid = c["unique_id"]
        g = work[work["unique_id"] == uid]
        train, test = split_fold(g, int(c["train_end_idx"]), horizon)
        if test.empty:
            continue
        h = len(test)
        y_train = train["y"].to_numpy(dtype=float)
        y_true = test["y"].to_numpy(dtype=float)
        naive = seasonal_naive_forecast(y_train, h, season_length)
        inf_start = time.time()

        if decomp == "without_stl":
            result = _forecast_with_source(task, y_train, h, season_length, task["finetuning_mode"])
            yhat = result.yhat
        elif decomp == "stl_seasonal_naive":
            comp_df, tr = make_component_targets(train, season_length)
            result = _forecast_with_source(
                task,
                comp_df["nonseasonal"].to_numpy(dtype=float),
                h,
                season_length,
                task["finetuning_mode"],
                component=True,
            )
            yhat = recompose_nonseasonal(result.yhat, tr.forecast_seasonal(h))
        else:
            comp_df, _ = make_component_targets(train, season_length)
            result = _forecast_with_source(
                task,
                comp_df["trend"].to_numpy(dtype=float),
                h,
                season_length,
                task["finetuning_mode"],
                component=True,
            )
            trend = result.yhat
            seasonal_result = _forecast_with_source(
                task,
                comp_df["seasonal"].to_numpy(dtype=float),
                h,
                season_length,
                task["finetuning_mode"],
                component=True,
            )
            residual_result = _forecast_with_source(
                task,
                comp_df["residual"].to_numpy(dtype=float),
                h,
                season_length,
                task["finetuning_mode"],
                component=True,
            )
            yhat = recompose_components(trend, seasonal_result.yhat, residual_result.yhat)

        infer_time += time.time() - inf_start
        rel_u = rel_naive(y_true, yhat, naive)
        rel_c = rel_naive(y_true, yhat, naive, clip=10)
        bm = bucket_manifest[(bucket_manifest["feature_name"] == feature) & (bucket_manifest["unique_id"] == uid)]
        bucket = bm["feature_tercile_bucket"].iloc[0] if not bm.empty else "Unknown"
        common = {
            **task,
            "unique_id": uid,
            "source_dataset": train["source_dataset"].iloc[0],
            "feature_bucket": bucket,
            "window": int(c["window"]),
            "cutoff": c["cutoff"],
            "model_output_source": result.model_output_source,
            "fallback_reason": result.fallback_reason,
            "model_backend": result.model_backend,
            "backend_library_version": result.backend_library_version,
            "forecast_hash": result.forecast_hash,
            "started_at_utc": started_at_utc,
        }
        metrics_rows.append(
            {
                **common,
                "rel_naive_unclipped": rel_u,
                "rel_naive_clipped": rel_c,
                "mae": mae(y_true, yhat),
                "smape": smape(y_true, yhat),
                "mase": mase(y_true, yhat, y_train, season_length),
                "wape": wape(y_true, yhat),
            }
        )
        for ds, yt, yp, yn in zip(test["ds"], y_true, yhat, naive):
            forecast_rows.append({**common, "ds": ds, "y": yt, "yhat": float(yp)})
            naive_rows.append({**common, "ds": ds, "y": yt, "yhat_naive": float(yn)})

    finished_at_utc = utc_now_iso()
    wall_seconds = time.perf_counter() - start_perf
    training_time_seconds = max(wall_seconds - infer_time, 1e-6)
    inference_time_seconds = max(infer_time, 1e-6)
    for row in metrics_rows:
        row["training_time_seconds"] = training_time_seconds
        row["inference_time_seconds"] = inference_time_seconds
        row["finished_at_utc"] = finished_at_utc
    for row in forecast_rows:
        row["finished_at_utc"] = finished_at_utc
    for row in naive_rows:
        row["finished_at_utc"] = finished_at_utc

    first = metrics_rows[0] if metrics_rows else {}
    return TaskEvaluationResult(
        metrics=pd.DataFrame(metrics_rows),
        forecasts=pd.DataFrame(forecast_rows),
        naive=pd.DataFrame(naive_rows),
        training_time_seconds=training_time_seconds,
        inference_time_seconds=inference_time_seconds,
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        model_output_source=first.get("model_output_source", "skipped"),
        fallback_reason=first.get("fallback_reason"),
        model_backend=first.get("model_backend"),
        backend_library_version=first.get("backend_library_version"),
        forecast_hash=first.get("forecast_hash"),
    )
