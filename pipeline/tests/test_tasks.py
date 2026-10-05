"""Task graph (protocol Section 9.2, D13, D15, D17, D18, D20)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.schemas import FEATURE_NAMES
from src.stages import tasks as tg

S0, EXTRA = 20260521, [20260522, 20260523]


def summary(degenerate_nonlinearity: bool) -> pd.DataFrame:
    rows = []
    for f in tg.FREQUENCIES:
        for feature in FEATURE_NAMES:
            populated = "Low,High" if degenerate_nonlinearity and feature == "feature_nonlinearity" else "Low,Medium,High"
            rows.append({"frequency": f, "feature_name": feature, "populated_buckets": populated})
    return pd.DataFrame(rows)


def samples_and_buckets(n_per_feature: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, brows = [], []
    for f in tg.FREQUENCIES:
        for k, feature in enumerate(FEATURE_NAMES):
            for i in range(n_per_feature):
                uid = f"{f}_S{(i + 3 * k) % 20:02d}"  # samples overlap across features, as for M3
                rows.append({"frequency": f, "feature_name": feature, "unique_id": uid})
                brows.append({"frequency": f, "feature_name": feature, "unique_id": uid,
                              "feature_tercile_bucket": ["Low", "Medium", "High"][i % 3]})
    return pd.DataFrame(rows), pd.DataFrame(brows)


def test_tuning_grid_has_90_studies():
    t = tg.tuning_tasks()
    assert len(t) == 90 and len({x.task_id for x in t}) == 90
    assert {x.shard for x in t} == {f"{fam}-{f}" for fam in ("ml", "neural", "transformer") for f in tg.FREQUENCIES}


@pytest.mark.parametrize("degenerate, tercile", [(True, 918), (False, 972)])
def test_evaluation_grid_matches_section_9_2(degenerate, tercile):
    samples, _ = samples_and_buckets()
    tasks = tg.evaluation_tasks(summary(degenerate), samples, S0, EXTRA)
    counts = tg.grid_counts(tasks)
    assert counts["cohort_global"] == 324 and counts["cohort_statistical_views"] == 108   # 432 cohort tasks
    assert counts["tercile"] == tercile
    assert counts["seed_check"] == 84
    assert len({t.task_id for t in tasks}) == len(tasks)
    seed_tasks = [t for t in tasks if t.priority == 2]
    assert {t.model for t in seed_tasks} == set(tg.STOCHASTIC_MODELS) and {t.seed for t in seed_tasks} == set(EXTRA)
    assert all(t.feature_name == tg.SEED_CHECK_FEATURE and t.scope == "cohort" for t in seed_tasks)
    assert [t.priority for t in tasks] == sorted(t.priority for t in tasks)  # cohort first, then tercile, then seeds
    assert {t.shard for t in tasks} == {f"{fam}-{f}" for fam in ("statistical", "ml", "neural", "transformer")
                                        for f in tg.FREQUENCIES}


def test_degenerate_nonlinearity_runs_with_its_populated_buckets_only():
    samples, _ = samples_and_buckets()
    tasks = tg.evaluation_tasks(summary(True), samples, S0, EXTRA)
    scopes = {t.scope for t in tasks if t.feature_name == "feature_nonlinearity" and t.priority == 1}
    assert scopes == {"low", "high"}


def test_statistical_chunks_cover_the_union_of_samples_exactly_once():
    samples, _ = samples_and_buckets()
    tasks = tg.evaluation_tasks(summary(True), samples, S0, EXTRA)
    for f in tg.FREQUENCIES:
        chunks = sorted({t.chunk for t in tasks if t.kind == "statistical" and t.frequency == f})
        covered = [uid for c in chunks for uid in tg.statistical_chunk_members(samples, f, c)]
        assert sorted(covered) == sorted(samples.loc[samples.frequency == f, "unique_id"].unique())
        assert len(covered) == len(set(covered))


def test_training_pools_are_the_sample_or_one_of_its_buckets():
    samples, buckets = samples_and_buckets()
    task = tg.Task("global", "monthly", "NHITS", "direct", None, "feature_arch_stat", "cohort", S0, 0)
    members, bucket_of = tg.global_task_members(task, samples, buckets)
    assert len(members) == 12 and set(bucket_of.values()) == {"Low", "Medium", "High"}
    low = tg.Task("global", "monthly", "NHITS", "direct", None, "feature_arch_stat", "low", S0, 1)
    low_members, low_buckets = tg.global_task_members(low, samples, buckets)
    assert len(low_members) == 4 and set(low_buckets.values()) == {"Low"} and set(low_members) <= set(members)


def test_pilot_feature_subset_gives_its_grid_and_needs_the_seed_check_feature():
    """Section 8.1: two features, main seed only; the grid follows the bundle's features."""
    pilot = ["feature_nonlinearity", "feature_evolving_seasonality"]
    samples, _ = samples_and_buckets()
    table = summary(True)
    table = table[table["feature_name"].isin(pilot)]
    tasks = tg.evaluation_tasks(table, samples[samples["feature_name"].isin(pilot)], S0, [])
    counts = tg.grid_counts(tasks)
    assert counts == {"cohort_global": 2 * 2 * 3 * 9, "cohort_statistical_views": 2 * 2 * 3 * 3,
                      "tercile": 2 * 3 * 9 * (3 + 2), "seed_check": 0, "statistical_chunks": 2 * 3 * 3}
    with pytest.raises(ValueError, match="seed check needs"):
        tg.evaluation_tasks(table[table["feature_name"] == "feature_nonlinearity"], samples, S0, EXTRA)


def test_selected_features_validates_and_keeps_canonical_order():
    from src.data.schemas import selected_features

    assert selected_features({}) == FEATURE_NAMES
    assert selected_features({"features": ["feature_arch_stat", "feature_non_normality"]}) == ["feature_non_normality",
                                                                                               "feature_arch_stat"]
    for bad in ([], ["feature_arch_stat", "feature_arch_stat"], ["feature_unknown"]):
        with pytest.raises(ValueError):
            selected_features({"features": bad})
