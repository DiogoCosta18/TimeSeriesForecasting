"""The six features of every series, on the history before the first test window.

Protocol D4/D5, defects C3 and F1. For a series of length n with W test windows of h
points, the features see only y[1 .. n - W*h]; nothing after that position is passed
to the computation at all.

Each feature either returns its value, is undefined for the series (FeatureUndefined:
too short, constant, degenerate) or fails (any other exception). Undefined and failed
features get the value NaN and a quality flag naming the reason ("undefined:<reason>",
"error:<type>: <message>"); no substitute value is ever written, and a series with any
flag other than "ok" is ineligible. There is no per-series time limit: eligibility must
not depend on the speed of the machine.
"""
from __future__ import annotations

import concurrent.futures as futures
import json
import logging
import math
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from src.data.schemas import FEATURE_NAMES
from src.utils import write_dataframe

from .arch import arch_feature
from .errors import FeatureUndefined
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
EXTRA_COLUMNS = ["seasonal_strength_first", "seasonal_strength_last", "seasonal_strength_slope"]
CACHE_FORMAT = "feature-cache/1"


def _evolving(y, season_length):
    return evolving_seasonality_feature(y, season_length)


FEATURE_FUNCTIONS: dict[str, Callable] = {
    "feature_non_normality": non_normality_feature,
    "feature_nonlinearity": nonlinearity_feature,
    "feature_spectral_entropy": lambda y, season_length: spectral_entropy_feature(y),
    "feature_evolving_seasonality": _evolving,
    "feature_structural_break_strength": structural_break_feature,
    "feature_arch_stat": arch_feature,
}
if list(FEATURE_FUNCTIONS) != FEATURE_NAMES:
    raise RuntimeError("FEATURE_FUNCTIONS must follow FEATURE_NAMES")


def compute_features_for_series(y, season_length: int) -> tuple[dict, dict]:
    """Values (NaN when not "ok") and quality flags of the six features of one history."""
    vals: dict[str, float] = {col: math.nan for col in EXTRA_COLUMNS}
    flags: dict[str, str] = {}
    for name, fn in FEATURE_FUNCTIONS.items():
        flag_name = QUALITY_FLAG_NAMES[name]
        vals[name] = math.nan
        try:
            out = fn(y, season_length)
        except FeatureUndefined as exc:
            flags[flag_name] = f"undefined:{exc.reason}"
            continue
        except Exception as exc:  # recorded as a failure, never replaced by a value
            flags[flag_name] = f"error:{type(exc).__name__}: {exc}"[:300]
            continue
        if name == "feature_evolving_seasonality":
            out, extras = out
            vals.update(extras)
        if not math.isfinite(out):
            flags[flag_name] = "error:non_finite_value"
            continue
        vals[name] = float(out)
        flags[flag_name] = "ok"
    return vals, flags


def history_end(series_length: int, horizon: int, n_windows: int) -> int:
    """Number of points before the first test window (1-based position of the last one)."""
    return series_length - n_windows * horizon


def _compute_one_series(payload: tuple) -> tuple[dict, dict]:
    uid, source_dataset, history, season_length, series_length, end = payload
    vals, flags = compute_features_for_series(history, int(season_length))
    meta = {"unique_id": uid, "source_dataset": source_dataset}
    row = {**meta, "series_length": int(series_length), "history_end_t": int(end), **vals}
    return row, {**meta, **flags}


def _series_payloads(df: pd.DataFrame, season_length: int, horizon: int, n_windows: int) -> list[tuple]:
    order = "t" if "t" in df.columns else "ds"
    payloads = []
    for uid, g in df.groupby("unique_id", sort=True):
        y = g.sort_values(order)["y"].to_numpy(dtype=float)
        end = history_end(len(y), horizon, n_windows)
        payloads.append((uid, g["source_dataset"].iloc[0], y[: max(end, 0)].copy(), int(season_length), len(y), end))
    return payloads


def _source_cache_name(frequency: str, source_dataset: str) -> str:
    safe = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in source_dataset)
    return f"{frequency}_{safe}_features_partial.parquet"


def _check_cache_meta(cache_dir: Path, meta: dict) -> None:
    """A cache is only reused for exactly the parameters it was written with."""
    path = cache_dir / "cache_meta.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != meta:
            raise ValueError(f"feature cache {cache_dir} was written with {stored}, not {meta}")
    else:
        path.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")


def compute_feature_table(
    df: pd.DataFrame,
    season_length: int,
    horizon: int,
    n_windows: int = 3,
    n_jobs: int = 1,
    cache_dir: Path | None = None,
    frequency: str = "unknown",
    logger: logging.Logger | None = None,
    status_callback: Callable[[dict], None] | None = None,
    checkpoint_every_series: int = 100,
    progress_log_every_series: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature values and quality flags of every series in ``df``, sorted by unique_id.

    ``cache_dir`` lets an interrupted computation resume: completed series are
    checkpointed per source and skipped on restart, provided the parameters match.
    """
    logger = logger or logging.getLogger(__name__)
    n_jobs = max(1, int(n_jobs))
    checkpoint_every_series = max(1, int(checkpoint_every_series))
    progress_log_every_series = max(1, int(progress_log_every_series))
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _check_cache_meta(cache_dir, {
            "format": CACHE_FORMAT, "frequency": frequency, "season_length": int(season_length),
            "horizon": int(horizon), "n_windows": int(n_windows),
        })

    logger.info("Starting feature computation for frequency=%s with n_jobs=%s", frequency, n_jobs)
    all_rows: list[pd.DataFrame] = []
    all_flags: list[pd.DataFrame] = []
    total_series = int(df["unique_id"].nunique())
    global_completed = 0
    started = time.time()

    for source_dataset, source_df in df.groupby("source_dataset", sort=True):
        source_payloads = _series_payloads(source_df, season_length, horizon, n_windows)
        cache_path = cache_dir / _source_cache_name(frequency, source_dataset) if cache_dir else None
        flag_cache_path = cache_path.with_name(cache_path.stem + "_quality_flags.parquet") if cache_path else None
        rows: list[dict] = []
        flag_rows: list[dict] = []
        if cache_path is not None and cache_path.exists():
            rows = pd.read_parquet(cache_path).to_dict("records")
            flag_rows = pd.read_parquet(flag_cache_path).to_dict("records")
        completed_ids = {str(r["unique_id"]) for r in rows}
        if completed_ids != {str(r["unique_id"]) for r in flag_rows}:
            raise ValueError(f"feature cache for {source_dataset} has values and flags for different series")
        pending_payloads = [p for p in source_payloads if str(p[0]) not in completed_ids]
        source_total = len(source_payloads)
        source_done = len(completed_ids)
        logger.info(
            "Feature cache for frequency=%s source=%s has %s/%s completed; pending=%s",
            frequency, source_dataset, source_done, source_total, len(pending_payloads),
        )

        def record_progress(current_uid: str) -> None:
            elapsed = time.time() - started
            eta = elapsed * (total_series - global_completed) / max(global_completed, 1)
            payload = {
                "stage": "compute_features",
                "frequency": frequency,
                "source_dataset": source_dataset,
                "current_unique_id": current_uid,
                "feature_completed_series": global_completed,
                "feature_total_series": total_series,
                "elapsed_seconds": round(elapsed, 1),
                "estimated_remaining_seconds": round(eta, 1),
            }
            logger.info(
                "feature_progress frequency=%s source=%s completed=%s/%s elapsed=%.1fs eta=%.1fs",
                frequency, source_dataset, global_completed, total_series, elapsed, eta,
            )
            if status_callback:
                status_callback(payload)

        def checkpoint() -> None:
            if cache_path:
                write_dataframe(cache_path, pd.DataFrame(rows))
                write_dataframe(flag_cache_path, pd.DataFrame(flag_rows))

        def record(row: dict, flag: dict) -> None:
            nonlocal global_completed, source_done
            rows.append(row)
            flag_rows.append(flag)
            global_completed += 1
            source_done += 1
            if source_done % progress_log_every_series == 0 or source_done == source_total:
                record_progress(str(row["unique_id"]))
            if source_done % checkpoint_every_series == 0 or source_done == source_total:
                checkpoint()

        global_completed += source_done
        if n_jobs == 1:
            for payload in pending_payloads:
                record(*_compute_one_series(payload))
        elif pending_payloads:
            with futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                for fut in futures.as_completed([executor.submit(_compute_one_series, p) for p in pending_payloads]):
                    record(*fut.result())  # a worker crash is an error, not a feature value
        checkpoint()
        all_rows.append(pd.DataFrame(rows))
        all_flags.append(pd.DataFrame(flag_rows))

    raw = pd.concat(all_rows, ignore_index=True).sort_values("unique_id", kind="mergesort").reset_index(drop=True)
    flags = pd.concat(all_flags, ignore_index=True).sort_values("unique_id", kind="mergesort").reset_index(drop=True)
    if raw["unique_id"].duplicated().any() or len(raw) != total_series:
        raise ValueError("feature table does not hold exactly one row per series")
    for name in FEATURE_NAMES:
        raw[name] = raw[name].astype(float)
    return raw, flags


def features_ok(flags: pd.DataFrame) -> pd.Series:
    """True where all six quality flags are "ok" (protocol D4: otherwise the series is ineligible)."""
    return flags[list(QUALITY_FLAG_NAMES.values())].eq("ok").all(axis=1)
