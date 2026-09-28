from __future__ import annotations

import pandas as pd


def make_cutoffs(df: pd.DataFrame, h: int, n_windows: int, step_size: int) -> pd.DataFrame:
    rows = []
    for uid, g in df.groupby("unique_id", sort=True):
        g = g.sort_values("ds")
        n = len(g)
        max_train_end = n - h
        for w in range(n_windows):
            train_end = max_train_end - (n_windows - 1 - w) * step_size
            if train_end <= max(3, h // 2):
                continue
            rows.append({"unique_id": uid, "cutoff": g["ds"].iloc[train_end - 1], "train_end_idx": train_end, "window": w})
    return pd.DataFrame(rows)


def split_fold(g: pd.DataFrame, train_end_idx: int, h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = g.sort_values("ds").reset_index(drop=True)
    return g.iloc[:train_end_idx].copy(), g.iloc[train_end_idx : train_end_idx + h].copy()

