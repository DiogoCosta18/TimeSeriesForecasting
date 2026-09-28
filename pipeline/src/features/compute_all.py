from __future__ import annotations

import concurrent.futures as futures
import logging
import os
import signal
import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from src.data.schemas import FEATURE_NAMES
from src.utils import write_dataframe

from .arch import arch_feature
from .mstl_features import evolving_seasonality_feature, non_normality_feature
from .nonlinear import nonlinearity_feature
from .spectral import spectral_entropy_feature
from .structural_breaks import structural_break_feature

QUALITY_FLAG_NAMES = {
    "feature_non_normality": "feature_quality_flag_non_normality",
    "feature_nonlinearity": "feature_quality_flag_nonlinearity",
    "feature_spectral_entropy": "feature_quality_flag_spectral_entropy",
    "feature_evolving_seasonality": "feature_quality_flag_evolving_seasonality",
    "feature_structural_break_strength": "feature_quality_flag_structural_break_strength",
    "feature_arch_stat": "feature_quality_flag_arch_stat",
}


def fallback_feature_values(reason: str) -> tuple[dict, dict]:
    vals = {
        "feature_non_normality": 0.0,
        "feature_nonlinearity": 0.0,
        "feature_spectral_entropy": 1.0,
        "feature_evolving_seasonality": 0.0,
        "feature_structural_break_strength": 0.0,
        "feature_arch_stat": 0.0,
    }
    flags = {flag: reason for flag in QUALITY_FLAG_NAMES.values()}
    return vals, flags


def compute_features_for_series(y, season_length: int) -> tuple[dict, dict]:
    vals: dict[str, float] = {}
    flags: dict[str, str] = {}
    vals["feature_non_normality"], flags["feature_quality_flag_non_normality"] = non_normality_feature(y, season_length)
    vals["feature_nonlinearity"], flags["feature_quality_flag_nonlinearity"] = nonlinearity_feature(y, season_length)
    vals["feature_spectral_entropy"], flags["feature_quality_flag_spectral_entropy"] = spectral_entropy_feature(y)
    evo, flag, extras = evolving_seasonality_feature(y, season_length)
    vals["feature_evolving_seasonality"] = evo
    vals.update(extras)
    flags["feature_quality_flag_evolving_seasonality"] = flag
    vals["feature_structural_break_strength"], flags["feature_quality_flag_structural_break_strength"] = structural_break_feature(y, season_length)
    vals["feature_arch_stat"], flags["feature_quality_flag_arch_stat"] = arch_feature(y, season_length)
    return vals, flags


def _timeout_handler(signum, frame):  # pragma: no cover - depends on Unix signal delivery
    raise TimeoutError("feature computation timed out")


def _compute_one_series(payload: tuple) -> tuple[dict, dict]:
    uid, source_dataset, y, season_length, horizon, trim_final_horizon, timeout_seconds = payload
    if trim_final_horizon and len(y) > horizon:
        y = y[:-horizon]
    previous_handler = None
    alarm_supported = hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")
    try:
        if alarm_supported and timeout_seconds and timeout_seconds > 0:
            previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(timeout_seconds))
        vals, flags = compute_features_for_series(y, int(season_length))
        reason = "ok"
    except TimeoutError:
        vals, flags = fallback_feature_values("fallback_timeout")
        reason = "fallback_timeout"
    except Exception as exc:
        vals, flags = fallback_feature_values(f"fallback_error:{exc.__class__.__name__}")
        reason = f"fallback_error:{exc.__class__.__name__}"
    finally:
        if alarm_supported:
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)
    meta = {"unique_id": uid, "source_dataset": source_dataset}
    return {**meta, **vals}, {**meta, **flags, "feature_quality_flag_worker": reason}


def _series_payloads(df: pd.DataFrame, season_length: int, horizon: int, trim_final_horizon: bool, timeout_seconds: int) -> list[tuple]:
    payloads = []
    for uid, g in df.groupby("unique_id", sort=True):
        g = g.sort_values("ds")
        payloads.append(
            (
                uid,
                g["source_dataset"].iloc[0],
                g["y"].to_numpy(dtype=float),
                int(season_length),
                int(horizon),
                bool(trim_final_horizon),
                int(timeout_seconds),
            )
        )
    return payloads


def _read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        csv = path.with_suffix(".csv")
        if csv.exists():
            return pd.read_csv(csv)
    return pd.DataFrame()


def _source_cache_name(frequency: str, source_dataset: str) -> str:
    safe = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in source_dataset)
    return f"{frequency}_{safe}_features_partial.parquet"


def compute_feature_table(
    df: pd.DataFrame,
    season_length: int,
    horizon: int,
    trim_final_horizon: bool = True,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
    frequency: str = "unknown",
    logger: logging.Logger | None = None,
    status_callback: Callable[[dict], None] | None = None,
    checkpoint_every_series: int = 100,
    progress_log_every_series: int = 25,
    timeout_seconds_per_series: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute all six features with cache/resume and process parallelism.

    Worker payloads are one small tuple per series to avoid copying large dataframes
    into each process. On Linux/Vast each worker applies a signal alarm per series.
    """

    logger = logger or logging.getLogger(__name__)
    n_jobs = max(1, int(n_jobs))
    checkpoint_every_series = max(1, int(checkpoint_every_series))
    progress_log_every_series = max(1, int(progress_log_every_series))
    timeout_seconds_per_series = max(1, int(timeout_seconds_per_series))
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting feature computation for frequency=%s with n_jobs=%s timeout_seconds_per_series=%s",
        frequency,
        n_jobs,
        timeout_seconds_per_series,
    )
    all_rows: list[pd.DataFrame] = []
    all_flags: list[pd.DataFrame] = []
    total_series = int(df["unique_id"].nunique())
    global_completed = 0
    started = time.time()

    for source_dataset, source_df in df.groupby("source_dataset", sort=True):
        source_payloads = _series_payloads(source_df, season_length, horizon, trim_final_horizon, timeout_seconds_per_series)
        cache_path = cache_dir / _source_cache_name(frequency, source_dataset) if cache_dir else None
        flag_cache_path = cache_path.with_name(cache_path.stem + "_quality_flags.parquet") if cache_path else None
        cached_rows = _read_cache(cache_path) if cache_path else pd.DataFrame()
        cached_flags = _read_cache(flag_cache_path) if flag_cache_path else pd.DataFrame()
        completed_ids = set(cached_rows["unique_id"].astype(str)) if not cached_rows.empty and "unique_id" in cached_rows else set()
        pending_payloads = [p for p in source_payloads if str(p[0]) not in completed_ids]
        source_total = len(source_payloads)
        source_done = len(completed_ids)
        rows = [] if cached_rows.empty else cached_rows.to_dict("records")
        flag_rows = [] if cached_flags.empty else cached_flags.to_dict("records")
        logger.info(
            "Feature cache for frequency=%s source=%s has %s/%s completed; pending=%s; n_jobs=%s",
            frequency,
            source_dataset,
            source_done,
            source_total,
            len(pending_payloads),
            n_jobs,
        )

        def record_progress(current_uid: str) -> None:
            elapsed = time.time() - started
            pct = 100.0 * max(global_completed, 1) / max(total_series, 1)
            eta = elapsed * (total_series - global_completed) / max(global_completed, 1)
            payload = {
                "stage": "compute_features",
                "frequency": frequency,
                "current_feature": "all_features",
                "current_unique_id": current_uid,
                "source_dataset": source_dataset,
                "feature_completed_series": global_completed,
                "feature_total_series": total_series,
                "feature_compute_n_jobs": n_jobs,
                "elapsed_seconds": round(elapsed, 1),
                "estimated_remaining_seconds": round(eta, 1),
                "percent_complete_estimate": round(pct, 2),
            }
            logger.info(
                "feature_progress frequency=%s source=%s feature=all_features completed=%s/%s percent=%.2f elapsed=%.1fs eta=%.1fs current_unique_id=%s n_jobs=%s",
                frequency,
                source_dataset,
                global_completed,
                total_series,
                pct,
                elapsed,
                eta,
                current_uid,
                n_jobs,
            )
            if status_callback:
                status_callback(payload)

        def checkpoint() -> None:
            if cache_path:
                write_dataframe(cache_path, pd.DataFrame(rows))
            if flag_cache_path:
                write_dataframe(flag_cache_path, pd.DataFrame(flag_rows))

        global_completed += source_done
        if pending_payloads:
            if n_jobs == 1:
                for payload in pending_payloads:
                    row, flag = _compute_one_series(payload)
                    rows.append(row)
                    flag_rows.append(flag)
                    global_completed += 1
                    source_done += 1
                    if source_done % progress_log_every_series == 0 or source_done == source_total:
                        record_progress(str(payload[0]))
                    if source_done % checkpoint_every_series == 0 or source_done == source_total:
                        checkpoint()
            else:
                with futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    future_to_uid = {executor.submit(_compute_one_series, payload): str(payload[0]) for payload in pending_payloads}
                    for fut in futures.as_completed(future_to_uid):
                        uid = future_to_uid[fut]
                        try:
                            row, flag = fut.result()
                        except Exception as exc:
                            vals, flags = fallback_feature_values(f"fallback_executor_error:{exc.__class__.__name__}")
                            row = {"unique_id": uid, "source_dataset": source_dataset, **vals}
                            flag = {"unique_id": uid, "source_dataset": source_dataset, **flags, "feature_quality_flag_worker": f"fallback_executor_error:{exc.__class__.__name__}"}
                        rows.append(row)
                        flag_rows.append(flag)
                        global_completed += 1
                        source_done += 1
                        if source_done % progress_log_every_series == 0 or source_done == source_total:
                            record_progress(uid)
                        if source_done % checkpoint_every_series == 0 or source_done == source_total:
                            checkpoint()
        else:
            record_progress("cache_complete")

        checkpoint()
        all_rows.append(pd.DataFrame(rows).drop_duplicates("unique_id", keep="last"))
        all_flags.append(pd.DataFrame(flag_rows).drop_duplicates("unique_id", keep="last"))

    raw = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    flags = pd.concat(all_flags, ignore_index=True) if all_flags else pd.DataFrame()
    for name in FEATURE_NAMES:
        if name not in raw:
            raw[name] = fallback_feature_values("fallback_missing")[0][name]
    return raw, flags

