from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import compute_all
from src.features.compute_all import compute_feature_table, fallback_feature_values
from src.training.run_experiment import _limit_candidate_series_for_feature_compute


def _df(n_series: int = 6, n_obs: int = 48) -> pd.DataFrame:
    rows = []
    for i in range(n_series):
        uid = f"s{i:03d}"
        source = "M3_Monthly" if i < n_series // 2 else "M4_Monthly"
        t = np.arange(n_obs)
        y = 10 + i + 0.1 * t + np.sin(2 * np.pi * t / 12)
        for ds, val in zip(pd.date_range("2000-01-01", periods=n_obs, freq="MS"), y):
            rows.append({"unique_id": uid, "source_dataset": source, "ds": ds, "y": val})
    return pd.DataFrame(rows)


def test_feature_compute_n_jobs_same_ids_and_columns(tmp_path):
    data = _df()
    raw1, flags1 = compute_feature_table(data, 12, 6, n_jobs=1, cache_dir=tmp_path / "one", frequency="monthly", progress_log_every_series=2)
    raw2, flags2 = compute_feature_table(data, 12, 6, n_jobs=2, cache_dir=tmp_path / "two", frequency="monthly", progress_log_every_series=2)
    assert set(raw1["unique_id"]) == set(raw2["unique_id"])
    assert set(raw1.columns) == set(raw2.columns)
    assert set(flags1["unique_id"]) == set(flags2["unique_id"])
    assert set(flags1.columns) == set(flags2.columns)


def test_feature_cache_resume_skips_completed_unique_ids(tmp_path, monkeypatch):
    data = _df(n_series=3)
    vals, flags = fallback_feature_values("cached")
    cached = pd.DataFrame([{"unique_id": "s000", "source_dataset": "M3_Monthly", **vals}])
    cached_flags = pd.DataFrame([{"unique_id": "s000", "source_dataset": "M3_Monthly", **flags}])
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached.to_parquet(cache_dir / "monthly_M3_Monthly_features_partial.parquet", index=False)
    cached_flags.to_parquet(cache_dir / "monthly_M3_Monthly_features_partial_quality_flags.parquet", index=False)

    original = compute_all.compute_features_for_series

    def fail_if_cached(y, season_length):
        if np.isclose(np.asarray(y)[0], 10.0):
            raise AssertionError("cached series was recomputed")
        return original(y, season_length)

    monkeypatch.setattr(compute_all, "compute_features_for_series", fail_if_cached)
    raw, _ = compute_feature_table(data, 12, 6, n_jobs=1, cache_dir=cache_dir, frequency="monthly")
    assert set(raw["unique_id"]) == {"s000", "s001", "s002"}


def test_failing_series_gets_fallback_flags(monkeypatch, tmp_path):
    data = _df(n_series=1)

    def fail(y, season_length):
        raise RuntimeError("boom")

    monkeypatch.setattr(compute_all, "compute_features_for_series", fail)
    raw, flags = compute_feature_table(data, 12, 6, n_jobs=1, cache_dir=tmp_path, frequency="monthly")
    assert raw.loc[0, "feature_spectral_entropy"] == 1.0
    assert "fallback_error" in flags.loc[0, "feature_quality_flag_worker"]


def test_pilot_candidate_limit_per_source():
    data = _df(n_series=20)

    class Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    limited = _limit_candidate_series_for_feature_compute(data, max_per_source=3, seed=123, frequency="monthly", logger=Logger())
    counts = limited[["unique_id", "source_dataset"]].drop_duplicates().groupby("source_dataset").size()
    assert counts.max() == 3
    assert limited["unique_id"].nunique() == 6

