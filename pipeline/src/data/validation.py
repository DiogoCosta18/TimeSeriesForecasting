"""Rolling-origin windows (protocol D8) for eligible series only (D3; test U2; defect M4)."""
from __future__ import annotations

import pandas as pd

from src.data.eligibility import IneligibleSeriesError


def make_cutoffs(df: pd.DataFrame, h: int, n_windows: int, step_size: int, *, min_history: int) -> pd.DataFrame:
    """Cutoffs of every series: window w trains on the first n - h - (n_windows - 1 - w) * step_size points.

    With step_size = h (protocol D8) the training set of window w ends at n - (3 - w)h.
    A series whose first training set holds fewer than ``min_history`` points (L of
    protocol D3) is refused, instead of having its early windows skipped silently.
    """
    if h < 1 or n_windows < 1 or step_size < 1:
        raise ValueError("h, n_windows and step_size must be positive")
    order = "t" if "t" in df.columns else "ds"
    rows, short = [], []
    for uid, g in df.groupby("unique_id", sort=True):
        g = g.sort_values(order)
        n = len(g)
        first_end = n - h - (n_windows - 1) * step_size
        if first_end < min_history:
            short.append(uid)
            continue
        for w in range(n_windows):
            train_end = first_end + w * step_size
            rows.append({"unique_id": uid, "cutoff": g["ds"].iloc[train_end - 1], "train_end_idx": train_end, "window": w})
    if short:
        raise IneligibleSeriesError(
            f"{len(short)} series have fewer than {min_history} points before the first test window, e.g. {short[:5]}"
        )
    return pd.DataFrame(rows, columns=["unique_id", "cutoff", "train_end_idx", "window"])


def split_fold(g: pd.DataFrame, train_end_idx: int, h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = g.sort_values("ds").reset_index(drop=True)
    return g.iloc[:train_end_idx].copy(), g.iloc[train_end_idx : train_end_idx + h].copy()
