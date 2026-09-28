from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schemas import FEATURE_NAMES


def make_tercile_bins(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    out = pd.Series("Medium", index=x.index, dtype="object")

    if x.isna().all():
        return out.astype(str)

    non_null = x.dropna()
    n_unique = int(non_null.nunique(dropna=True))

    if n_unique < 3:
        if n_unique == 2:
            unique_vals = np.sort(non_null.unique())
            low_val, high_val = unique_vals[0], unique_vals[-1]
            out.loc[x == low_val] = "Low"
            out.loc[x == high_val] = "High"
        return out.astype(str)

    try:
        binned = pd.qcut(non_null, q=3, labels=["Low", "Medium", "High"], duplicates="drop")
        out.loc[non_null.index] = binned.astype(str)
        return out.astype(str)
    except Exception:
        pass

    try:
        binned = pd.qcut(non_null, q=3, duplicates="drop")
        n_bins = int(len(binned.cat.categories))
        if n_bins <= 1:
            out.loc[non_null.index] = "Medium"
        elif n_bins == 2:
            codes = binned.cat.codes
            mapped = pd.Series(np.where(codes == 0, "Low", "High"), index=non_null.index)
            out.loc[non_null.index] = mapped
        else:
            codes = binned.cat.codes
            mapping = {0: "Low", 1: "Medium", 2: "High"}
            mapped = pd.Series(codes, index=non_null.index).map(mapping).fillna("Medium")
            out.loc[non_null.index] = mapped
        return out.astype(str)
    except Exception:
        pass

    # Final safety net for degenerate distributions with repeated values.
    ranks = non_null.rank(method="average", pct=True)
    out.loc[non_null.index] = np.select(
        [ranks <= (1.0 / 3.0), ranks <= (2.0 / 3.0)],
        ["Low", "Medium"],
        default="High",
    )
    return out.astype(str)


def fit_transform_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = raw.copy()
    scaler = {}
    for name in FEATURE_NAMES:
        x = pd.to_numeric(out[name], errors="coerce")
        lo, hi = x.quantile([0.01, 0.99]).tolist()
        med = float(x.median()) if np.isfinite(x.median()) else 0.0
        clipped_no_fill = x.clip(lo, hi)
        clipped = clipped_no_fill.fillna(med)
        mean = float(clipped.mean())
        std = float(clipped.std(ddof=0)) or 1.0
        out[f"{name}_zscore"] = (clipped - mean) / std
        q1, q2 = clipped.quantile([1 / 3, 2 / 3]).tolist()
        out[f"{name}_bin"] = make_tercile_bins(clipped_no_fill)
        scaler[name] = {"winsor_p01": float(lo), "winsor_p99": float(hi), "median": med, "mean": mean, "std": std, "tercile_q1": float(q1), "tercile_q2": float(q2)}
    return out, scaler

