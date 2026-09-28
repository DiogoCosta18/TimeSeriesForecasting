import numpy as np

from src.features.compute_all import compute_features_for_series


def test_constant_short_missing_series_do_not_crash():
    for y in [np.ones(20), np.array([1.0, np.nan, 2.0]), np.arange(5, dtype=float)]:
        vals, flags = compute_features_for_series(y, season_length=12)
        assert "feature_non_normality" in vals
        assert "feature_arch_stat" in vals
        assert all(np.isfinite(v) for v in vals.values())
        assert flags

