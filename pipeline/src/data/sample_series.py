from __future__ import annotations

import numpy as np
import pandas as pd


def representative_sample(
    features: pd.DataFrame,
    feature_name: str,
    frequency: str,
    requested_n: int,
    n_strata: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for source, g in features.groupby("source_dataset", sort=True):
        g = g.dropna(subset=[feature_name]).copy()
        available = len(g)
        take_n = min(requested_n, available)
        if take_n == 0:
            continue
        g = g.sort_values([feature_name, "unique_id"]).reset_index(drop=True)
        g["feature_rank"] = np.arange(1, available + 1)
        g["feature_quantile"] = (g["feature_rank"] - 0.5) / max(available, 1)
        strata = min(n_strata, available)
        g["selection_stratum"] = pd.qcut(g["feature_rank"], q=strata, labels=False, duplicates="drop").astype(int)
        stratum_ids = sorted(g["selection_stratum"].unique())
        quota = {s: take_n // len(stratum_ids) for s in stratum_ids}
        for s in stratum_ids[: take_n % len(stratum_ids)]:
            quota[s] += 1
        selected = []
        remaining = take_n
        for s in stratum_ids:
            sg = g[g["selection_stratum"] == s]
            k = min(quota[s], len(sg))
            if k:
                selected.append(sg.sample(n=k, random_state=int(rng.integers(0, 2**31 - 1))))
                remaining -= k
        if remaining > 0:
            used = pd.concat(selected)["unique_id"] if selected else pd.Series([], dtype=str)
            pool = g[~g["unique_id"].isin(set(used))]
            selected.append(pool.sample(n=min(remaining, len(pool)), random_state=int(rng.integers(0, 2**31 - 1))))
        sel = pd.concat(selected).drop_duplicates("unique_id").head(take_n)
        sel = sel.assign(
            feature_name=feature_name,
            frequency=frequency,
            requested_n=requested_n,
            available_n=available,
            selected_n=len(sel),
            selection_seed=seed,
            feature_value=sel[feature_name].astype(float),
        )
        rows.append(sel)
    cols = [
        "feature_name",
        "frequency",
        "source_dataset",
        "requested_n",
        "available_n",
        "selected_n",
        "unique_id",
        "feature_value",
        "feature_rank",
        "feature_quantile",
        "selection_stratum",
        "selection_seed",
    ]
    return pd.concat(rows, ignore_index=True)[cols] if rows else pd.DataFrame(columns=cols)


def assign_feature_buckets(sample_manifest: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (feature, frequency), g in sample_manifest.groupby(["feature_name", "frequency"], sort=True):
        values = g["feature_value"].astype(float)
        q1, q2 = values.quantile([1 / 3, 2 / 3]).tolist()
        def bucket(v: float) -> str:
            if v <= q1:
                return "Low"
            if v <= q2:
                return "Medium"
            return "High"
        out = g[["feature_name", "frequency", "unique_id", "source_dataset", "feature_value"]].copy()
        out["feature_tercile_bucket"] = out["feature_value"].map(bucket)
        out["bucket_lower_bound"] = out["feature_tercile_bucket"].map({"Low": values.min(), "Medium": q1, "High": q2})
        out["bucket_upper_bound"] = out["feature_tercile_bucket"].map({"Low": q1, "Medium": q2, "High": values.max()})
        records.append(out)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

