from __future__ import annotations

import pandas as pd


def regularize_global_training_frame(df_train: pd.DataFrame, freq_str: str) -> pd.DataFrame:
    """Return a dense per-series training frame for global forecasting libraries.

    The benchmark evaluates forecasts by series order, not by the future timestamp values
    emitted by the backend. Some source datasets mix timestamp conventions or contain
    gaps under a single pandas frequency, so global libraries reject otherwise valid
    folds. A synthetic regular calendar preserves each series' ordered target history
    while satisfying backend frequency validation.
    """
    parts: list[pd.DataFrame] = []
    work = df_train[["unique_id", "ds", "y"]].copy()
    work["ds"] = pd.to_datetime(work["ds"])
    work["y"] = pd.to_numeric(work["y"], errors="coerce")
    work = work.dropna(subset=["unique_id", "ds", "y"])
    work = work.groupby(["unique_id", "ds"], as_index=False, sort=False)["y"].mean()

    for uid, g in work.groupby("unique_id", sort=False):
        ordered = g.sort_values("ds", kind="mergesort").reset_index(drop=True)
        if ordered.empty:
            continue
        ordered["ds"] = pd.date_range("2000-01-01", periods=len(ordered), freq=freq_str)
        ordered["unique_id"] = uid
        parts.append(ordered[["unique_id", "ds", "y"]])

    if not parts:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])
    return pd.concat(parts, ignore_index=True)
