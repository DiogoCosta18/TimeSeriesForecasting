"""Feature samples, tuning set and buckets (protocol D6, D7, D13; tests U7, U8)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.sample_series import (
    InsufficientPoolError,
    feature_sample,
    tercile_buckets,
    tuning_set,
)

S0 = 20260521


def pool(sizes: dict[str, int], seed: int = 0, ties: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for source, n in sizes.items():
        values = rng.normal(size=n)
        if ties:
            values = np.where(values < 0.15, 0.0, values)  # a floored feature, like nonlinearity
        parts.append(pd.DataFrame({
            "unique_id": [f"{source}_S{i:05d}" for i in range(n)],
            "source_dataset": source,
            "feature_nonlinearity": values,
            "feature_arch_stat": values,  # same values, different feature
        }))
    return pd.concat(parts, ignore_index=True)


POOL = {"M3_Monthly": 1039, "M4_Monthly": 3000}


def test_u7_sampling_is_deterministic_and_fills_every_stratum_quota():
    eligible = pool(POOL)
    a = feature_sample(eligible, "feature_nonlinearity", "monthly", 12, S0)
    b = feature_sample(eligible, "feature_nonlinearity", "monthly", 12, S0)
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby("source_dataset").size().to_dict() == {"M3_Monthly": 600, "M4_Monthly": 600}
    assert a["unique_id"].is_unique
    per_stratum = a.groupby(["source_dataset", "selection_stratum"]).size()
    assert len(per_stratum) == 60 and (per_stratum == 20).all()
    for source, g in a.groupby("source_dataset"):
        sizes = g.drop_duplicates("selection_stratum")["stratum_size"]
        assert sizes.max() - sizes.min() <= 1 and sizes.sum() == POOL[source]


def test_u7_ranks_quantiles_and_strata_follow_the_values():
    eligible = pool(POOL, ties=True)
    s = feature_sample(eligible, "feature_nonlinearity", "monthly", 12, S0)
    for source, g in s.groupby("source_dataset"):
        ranked = eligible[eligible["source_dataset"] == source].sort_values(["feature_nonlinearity", "unique_id"], kind="mergesort")
        rank_of = dict(zip(ranked["unique_id"], range(1, len(ranked) + 1)))
        assert (g["feature_rank"] == g["unique_id"].map(rank_of)).all()  # ties broken by unique_id
        np.testing.assert_allclose(g["feature_quantile"], (g["feature_rank"] - 0.5) / len(ranked))
        assert g.sort_values("feature_rank")["selection_stratum"].is_monotonic_increasing


def test_u7_each_feature_uses_its_own_seed():
    eligible = pool(POOL)
    a = feature_sample(eligible, "feature_nonlinearity", "monthly", 12, S0)
    b = feature_sample(eligible, "feature_arch_stat", "monthly", 12, S0)
    assert set(a["selection_seed"]).isdisjoint(set(b["selection_seed"]))
    assert set(a["unique_id"]) != set(b["unique_id"])  # identical values, different draws
    other_source_seeds = a.drop_duplicates("source_dataset")["selection_seed"].tolist()
    assert len(set(other_source_seeds)) == 2  # one stream per source as well


def test_u7_a_pool_smaller_than_the_quota_stops_with_a_decision_needed():
    with pytest.raises(InsufficientPoolError, match="599 eligible series < 600"):
        feature_sample(pool({"M3_Quarterly": 599}), "feature_nonlinearity", "quarterly", 4, S0)


def test_sampling_refuses_missing_feature_values():
    eligible = pool(POOL)
    eligible.loc[3, "feature_nonlinearity"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        feature_sample(eligible, "feature_nonlinearity", "monthly", 12, S0)


def test_d13_tuning_set_is_a_fixed_draw_of_600_per_source():
    eligible = pool(POOL)[["unique_id", "source_dataset"]]
    a = tuning_set(eligible, "monthly", 12, S0)
    pd.testing.assert_frame_equal(a, tuning_set(eligible, "monthly", 12, S0))
    assert a.groupby("source_dataset").size().to_dict() == {"M3_Monthly": 600, "M4_Monthly": 600}
    assert set(a["unique_id"]) <= set(eligible["unique_id"]) and a["unique_id"].is_unique
    with pytest.raises(InsufficientPoolError, match="599 eligible series < 600"):
        tuning_set(pool({"M3_Quarterly": 599})[["unique_id", "source_dataset"]], "quarterly", 4, S0)


# --- U8: buckets -----------------------------------------------------------------------------

def _sample(values) -> pd.DataFrame:
    return pd.DataFrame({
        "feature_name": "feature_arch_stat", "frequency": "monthly",
        "unique_id": [f"S{i:05d}" for i in range(len(values))], "feature_value": values,
    })


@pytest.mark.parametrize("n", [1200, 1201, 1202, 9])
def test_u8_tercile_sizes_differ_by_at_most_one(n):
    buckets, summary = tercile_buckets(_sample(np.random.default_rng(n).normal(size=n)))
    sizes = [summary["size_Low"], summary["size_Medium"], summary["size_High"]]
    assert sum(sizes) == n and max(sizes) - min(sizes) <= 1
    assert not summary["degenerate"]
    low, high = buckets[buckets.feature_tercile_bucket == "Low"], buckets[buckets.feature_tercile_bucket == "High"]
    assert low["feature_value"].max() < high["feature_value"].min()


def test_u8_support_guard_flags_a_floored_feature():
    values = np.random.default_rng(1).normal(size=1200)
    values = np.where(values < 0.15, 0.0, values)  # ~56% zeros
    buckets, summary = tercile_buckets(_sample(values))
    assert summary["size_Low"] == int((values == 0).sum())  # tied zeros share one bucket
    assert summary["degenerate"] and summary["share_Medium"] < 0.20


def test_u8_an_empty_bucket_is_degenerate():
    values = np.r_[np.zeros(840), np.linspace(1, 2, 360)]  # 70% ties: Medium is empty
    _, summary = tercile_buckets(_sample(values))
    assert summary["size_Medium"] == 0 and summary["populated_buckets"] == "Low,High"
    assert summary["degenerate"]


def test_buckets_take_one_feature_and_frequency_at_a_time():
    two = pd.concat([_sample([1.0, 2.0, 3.0]), _sample([1.0, 2.0, 3.0]).assign(frequency="quarterly")])
    with pytest.raises(ValueError, match="one feature sample"):
        tercile_buckets(two)
