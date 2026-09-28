from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schemas import FEATURE_NAMES
from src.features.scaling import fit_transform_features, make_tercile_bins


ALLOWED_LABELS = {"Low", "Medium", "High"}


def _raw_with_target(values: list[float] | np.ndarray, target: str = "feature_non_normality") -> pd.DataFrame:
    n = len(values)
    base = np.linspace(0.0, 1.0, n)
    data = {name: base.copy() for name in FEATURE_NAMES}
    data[target] = np.asarray(values, dtype=float)
    return pd.DataFrame(data)


def test_make_tercile_bins_constant_values_are_medium():
    s = pd.Series(np.zeros(24, dtype=float))
    bins = make_tercile_bins(s)
    assert set(bins.unique()) == {"Medium"}


def test_make_tercile_bins_two_unique_values_safe_labels():
    s = pd.Series([0.0] * 12 + [1.0] * 12)
    bins = make_tercile_bins(s)
    assert set(bins.unique()).issubset(ALLOWED_LABELS)
    assert set(bins.unique()) in ({"Medium"}, {"Low", "High"})


def test_make_tercile_bins_many_duplicate_zeros_with_few_positives():
    s = pd.Series([0.0] * 40 + [1.0] * 3 + [2.0] * 3)
    bins = make_tercile_bins(s)
    assert bins.notna().all()
    assert set(bins.unique()).issubset(ALLOWED_LABELS)


def test_make_tercile_bins_all_nan_returns_medium():
    s = pd.Series([np.nan] * 20)
    bins = make_tercile_bins(s)
    assert set(bins.unique()) == {"Medium"}


def test_make_tercile_bins_continuous_has_three_levels():
    s = pd.Series(np.arange(1, 61, dtype=float))
    bins = make_tercile_bins(s)
    assert set(bins.unique()) == ALLOWED_LABELS


def test_fit_transform_features_no_crash_with_duplicate_quantiles():
    raw = _raw_with_target([0.0] * 50 + [1.0] * 2 + [2.0] * 2)
    out, scaler = fit_transform_features(raw)
    assert "feature_non_normality_bin" in out.columns
    assert set(out["feature_non_normality_bin"].unique()).issubset(ALLOWED_LABELS)
    assert "feature_non_normality" in scaler
