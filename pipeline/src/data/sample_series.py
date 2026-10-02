"""Feature samples, tuning set and buckets (protocol D6, D7, D13; tests U4, U7, U8; gates G11, G12).

Everything here draws only from eligible series and is deterministic given the base
seed s0 (configuration ``random_seed``). Random streams are numpy Generators seeded
with a tuple, so every (purpose, frequency, source) has its own stream:
  feature k:   (s0 + k, m, source)       -- "seed = s0 + k" of protocol Section 3.4
  tuning set:  (s0, TUNING_STREAM, m, source)
with m the season length and source 3 for M3, 4 for M4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schemas import FEATURE_NAMES

SOURCE_CODE = {"M3": 3, "M4": 4}
TUNING_STREAM = 1000
BUCKETS = ("Low", "Medium", "High")
MIN_BUCKET_SHARE = 0.20


class InsufficientPoolError(ValueError):
    """An eligible pool is too small to fill the protocol's quotas; a decision is needed."""


def _source_code(source_dataset: str) -> int:
    return SOURCE_CODE[str(source_dataset).split("_", 1)[0]]


def feature_seed(base_seed: int, feature: str, season_length: int, source_dataset: str) -> tuple[int, int, int]:
    return (int(base_seed) + FEATURE_NAMES.index(feature), int(season_length), _source_code(source_dataset))


def feature_sample(
    eligible: pd.DataFrame,
    feature: str,
    frequency: str,
    season_length: int,
    base_seed: int,
    n_per_source: int = 600,
    n_strata: int = 30,
) -> pd.DataFrame:
    """Stratified quantile sample of one feature and frequency (protocol Section 3.4, D6).

    ``eligible`` holds unique_id, source_dataset and the feature value of every eligible
    series of the frequency. Per source: rank by value (ties by unique_id), cut the
    ranking into ``n_strata`` equal-count strata, draw n_per_source / n_strata per
    stratum without replacement, and record stratum, rank and quantile.
    """
    if n_per_source % n_strata:
        raise ValueError("n_per_source must be a multiple of n_strata")
    quota = n_per_source // n_strata
    parts = []
    for source, pool in eligible[["unique_id", "source_dataset", feature]].groupby("source_dataset", sort=True):
        values = pool[feature].astype(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{feature} {source}: non-finite values in the eligible pool")
        ranked = pool.assign(feature_value=values).sort_values(["feature_value", "unique_id"], kind="mergesort")
        n = len(ranked)
        if n < n_per_source:
            raise InsufficientPoolError(f"{feature} {frequency} {source}: {n} eligible series < {n_per_source}")
        ranked = ranked.reset_index(drop=True)
        ranked["feature_rank"] = np.arange(1, n + 1)
        ranked["feature_quantile"] = (ranked["feature_rank"] - 0.5) / n
        strata = np.array_split(np.arange(n), n_strata)
        ranked["selection_stratum"] = np.repeat(np.arange(n_strata), [len(s) for s in strata])
        seed = feature_seed(base_seed, feature, season_length, source)
        rng = np.random.default_rng(seed)
        chosen = np.concatenate([np.sort(rng.choice(members, size=quota, replace=False)) for members in strata])
        sel = ranked.iloc[chosen].copy()
        sel["stratum_size"] = sel["selection_stratum"].map({i: len(s) for i, s in enumerate(strata)})
        sel["pool_size"] = n
        sel["selection_seed"] = str(seed)
        parts.append(sel)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "feature_name", feature)
    out.insert(1, "frequency", frequency)
    cols = ["feature_name", "frequency", "source_dataset", "unique_id", "feature_value", "feature_rank",
            "feature_quantile", "selection_stratum", "stratum_size", "pool_size", "selection_seed"]
    return out[cols]


def tuning_set(eligible_ids: pd.DataFrame, frequency: str, season_length: int, base_seed: int, n_per_source: int = 600) -> pd.DataFrame:
    """Fixed simple random draw of n_per_source eligible series per source (protocol D13)."""
    parts = []
    for source, pool in eligible_ids.groupby("source_dataset", sort=True):
        ids = np.sort(pool["unique_id"].astype(str).to_numpy())
        if len(ids) < n_per_source:
            raise InsufficientPoolError(f"tuning set {frequency} {source}: {len(ids)} eligible series < {n_per_source}")
        seed = (int(base_seed), TUNING_STREAM, int(season_length), _source_code(source))
        chosen = np.sort(np.random.default_rng(seed).choice(ids, size=n_per_source, replace=False))
        parts.append(pd.DataFrame({"frequency": frequency, "source_dataset": source, "unique_id": chosen, "selection_seed": str(seed)}))
    return pd.concat(parts, ignore_index=True)


def tercile_buckets(sample: pd.DataFrame, min_share: float = MIN_BUCKET_SHARE) -> tuple[pd.DataFrame, dict]:
    """Low / Medium / High within one feature sample and frequency (protocol Section 3.5, D7).

    The cut points are the empirical 1/3 and 2/3 quantiles: with the values sorted (ties
    by unique_id), t1 is the value at position round(n/3) and t2 at round(2n/3); a series
    is Low if its value <= t1, Medium if t1 < value <= t2, High otherwise. Without ties the
    sizes differ by at most one; tied values always share a bucket. The feature is
    degenerate for this frequency if any of the three buckets (an empty one included)
    holds less than 20% of the sample: it is then run with its populated buckets only.
    """
    if sample[["feature_name", "frequency"]].drop_duplicates().shape[0] != 1:
        raise ValueError("tercile_buckets takes one feature sample of one frequency")
    ordered = sample.sort_values(["feature_value", "unique_id"], kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    values = ordered["feature_value"].to_numpy(dtype=float)
    t1, t2 = values[round(n / 3) - 1], values[round(2 * n / 3) - 1]
    ordered["feature_tercile_bucket"] = np.where(values <= t1, "Low", np.where(values <= t2, "Medium", "High"))
    ordered["tercile_cut_1"], ordered["tercile_cut_2"] = t1, t2
    sizes = ordered["feature_tercile_bucket"].value_counts().reindex(BUCKETS, fill_value=0)
    shares = sizes / n
    summary = {
        "feature_name": ordered["feature_name"].iloc[0],
        "frequency": ordered["frequency"].iloc[0],
        "n": n,
        "tercile_cut_1": float(t1),
        "tercile_cut_2": float(t2),
        **{f"size_{b}": int(sizes[b]) for b in BUCKETS},
        **{f"share_{b}": float(shares[b]) for b in BUCKETS},
        "degenerate": bool((shares < min_share).any()),
        "populated_buckets": ",".join(b for b in BUCKETS if sizes[b] > 0),
    }
    return ordered, summary


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

