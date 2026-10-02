from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import compute_all
from src.features.compute_all import compute_feature_table
from src.training.run_experiment import _limit_candidate_series_for_feature_compute


def _df(n_series: int = 6, n_obs: int = 120) -> pd.DataFrame:
    rows = []
    for i in range(n_series):
        uid = f"s{i:03d}"
        source = "M3_Monthly" if i < n_series // 2 else "M4_Monthly"
        t = np.arange(n_obs)
        y = 10 + i + 0.1 * t + np.sin(2 * np.pi * t / 12) + 0.3 * np.cos(0.7 * t * (i + 1))
        for pos, val in enumerate(y, start=1):
            rows.append({"unique_id": uid, "source_dataset": source, "t": pos, "ds": pos, "y": val})
    return pd.DataFrame(rows)


def test_feature_values_identical_for_any_number_of_workers(tmp_path):
    data = _df()
    raw1, flags1 = compute_feature_table(data, 12, 18, n_jobs=1, cache_dir=tmp_path / "one", frequency="monthly")
    raw2, flags2 = compute_feature_table(data, 12, 18, n_jobs=2, cache_dir=tmp_path / "two", frequency="monthly")
    pd.testing.assert_frame_equal(raw1, raw2)
    pd.testing.assert_frame_equal(flags1, flags2)
    assert raw1["unique_id"].tolist() == sorted(raw1["unique_id"])


def test_feature_cache_resume_skips_completed_series(tmp_path, monkeypatch):
    data = _df(n_series=4)
    cache_dir = tmp_path / "cache"
    raw, flags = compute_feature_table(data, 12, 18, cache_dir=cache_dir, frequency="monthly")

    def fail(payload):
        raise AssertionError(f"{payload[0]} was recomputed")

    monkeypatch.setattr(compute_all, "_compute_one_series", fail)
    raw_resumed, flags_resumed = compute_feature_table(data, 12, 18, cache_dir=cache_dir, frequency="monthly")
    pd.testing.assert_frame_equal(raw, raw_resumed)
    pd.testing.assert_frame_equal(flags, flags_resumed)


def test_feature_cache_refuses_other_parameters(tmp_path):
    data = _df(n_series=2)
    compute_feature_table(data, 12, 18, cache_dir=tmp_path, frequency="monthly")
    with pytest.raises(ValueError, match="was written with"):
        compute_feature_table(data, 12, 18, n_windows=2, cache_dir=tmp_path, frequency="monthly")


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
