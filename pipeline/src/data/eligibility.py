"""Eligibility of every series in the pool (protocol D3, D4; tests U2, U4; gate G8).

A series is eligible when (D3) its history before the first test window holds at
least L = 3m + h points (54 monthly, 20 quarterly) and (D4) all six features are
"ok" on that history. Features are computed only for series that pass the length
rule; the others carry the flag "not_computed:history_shorter_than_L".
"""
from __future__ import annotations

import pandas as pd

from src.data.schemas import FEATURE_NAMES
from src.features.compute_all import QUALITY_FLAG_NAMES, history_end

NOT_COMPUTED = "not_computed:history_shorter_than_L"


class IneligibleSeriesError(ValueError):
    """An ineligible series reached a stage that accepts only eligible series."""


def eligibility_length(season_length: int, horizon: int) -> int:
    """L = 3m + h (protocol D3): 54 for monthly (m=12, h=18), 20 for quarterly (m=4, h=8)."""
    return 3 * int(season_length) + int(horizon)


def length_table(df: pd.DataFrame, horizon: int, n_windows: int) -> pd.DataFrame:
    """unique_id, source_dataset, series_length and history_length (points before the first test window)."""
    g = df.groupby("unique_id", sort=True)
    out = pd.DataFrame({
        "source_dataset": g["source_dataset"].first(),
        "series_length": g["y"].size().astype("int64"),
    }).reset_index()
    out["history_length"] = [history_end(n, horizon, n_windows) for n in out["series_length"]]
    return out


def eligibility_table(
    lengths: pd.DataFrame,
    feature_flags: pd.DataFrame,
    season_length: int,
    horizon: int,
) -> pd.DataFrame:
    """One row per series of the pool: length rule, the six feature flags, eligible, reason.

    ``feature_flags`` must hold exactly the series that pass the length rule.
    """
    L = eligibility_length(season_length, horizon)
    out = lengths.copy()
    out["eligibility_length"] = L
    out["length_eligible"] = out["history_length"] >= L
    computed = set(out.loc[out["length_eligible"], "unique_id"])
    if set(feature_flags["unique_id"]) != computed or feature_flags["unique_id"].duplicated().any():
        raise ValueError("feature flags must be given exactly once for every series that passes the length rule")
    flag_cols = [QUALITY_FLAG_NAMES[name] for name in FEATURE_NAMES]
    out = out.merge(feature_flags[["unique_id", *flag_cols]], on="unique_id", how="left", validate="one_to_one")
    out[flag_cols] = out[flag_cols].fillna(NOT_COMPUTED)
    out["features_ok"] = out[flag_cols].eq("ok").all(axis=1)
    out["eligible"] = out["length_eligible"] & out["features_ok"]

    def reason(row) -> str:
        if not row["length_eligible"]:
            return f"history_length {row['history_length']} < L {L}"
        failing = [f"{name}={row[col]}" for name, col in zip(FEATURE_NAMES, flag_cols) if row[col] != "ok"]
        return "; ".join(failing)

    out["ineligible_reason"] = [("" if ok else reason(row)) for ok, (_, row) in zip(out["eligible"], out.iterrows())]
    return out.sort_values("unique_id", kind="mergesort").reset_index(drop=True)


def require_eligible(table: pd.DataFrame, ids) -> None:
    """Raise unless every id is an eligible series of ``table`` (used by sampling and cutoffs)."""
    eligible = set(table.loc[table["eligible"], "unique_id"])
    bad = sorted(set(ids) - eligible)
    if bad:
        raise IneligibleSeriesError(f"{len(bad)} ineligible or unknown series, e.g. {bad[:5]}")
