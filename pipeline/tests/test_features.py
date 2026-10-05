"""Features without fallbacks, on the history before the first test window.

Protocol D4, D5, D9; tests U1, U3, U5; defects C3, F1, F2. Series are synthetic and
built here; the pipeline itself never generates data.
"""
from __future__ import annotations

import ast
import inspect
import math

import numpy as np
import pandas as pd
import pytest

from src.data.schemas import FEATURE_NAMES
from src.data.validation import split_fold
from src.features import arch, compute_all, mstl_features, nonlinear, spectral, structural_breaks
from src.features.compute_all import compute_feature_table, compute_features_for_series, features_ok, history_end
from src.features.errors import DecompositionError, FeatureUndefined
from src.features.mstl_features import decompose_series
from src.forecast.targets import component_targets, seasonal_continuation

ELIGIBILITY_LENGTH = {12: 54, 4: 20}  # protocol D3: L = 3m + h


def synthetic(n: int, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 100 + 0.3 * t + 8 * np.sin(2 * np.pi * t / m + rng.uniform(0, 2 * np.pi)) + rng.normal(0, 2, n)


def frame(series: dict[str, np.ndarray], source: str = "M4_Monthly") -> pd.DataFrame:
    rows = [
        {"unique_id": uid, "source_dataset": source, "t": t, "ds": t, "y": float(v)}
        for uid, values in series.items() for t, v in enumerate(values, start=1)
    ]
    return pd.DataFrame(rows)


# --- U3 and D5: defined at exactly the eligibility length; guards as in the protocol ----------

@pytest.mark.parametrize("m", [12, 4])
def test_u3_every_feature_is_defined_at_exactly_the_eligibility_length(m):
    for seed in range(25):
        vals, flags = compute_features_for_series(synthetic(ELIGIBILITY_LENGTH[m], m, seed), m)
        assert set(flags.values()) == {"ok"}, (seed, flags)
        assert all(math.isfinite(vals[name]) for name in FEATURE_NAMES)


@pytest.mark.parametrize("fn", [nonlinear.nonlinearity_feature, arch.arch_feature, structural_breaks.structural_break_feature])
def test_d5_guards_are_48_monthly_and_20_quarterly(fn):
    for m, guard in [(12, 48), (4, 20)]:
        fn(synthetic(guard, m, 1), m)
        with pytest.raises(FeatureUndefined, match="too_short"):
            fn(synthetic(guard - 1, m, 1), m)


# --- F1: no substitute values -----------------------------------------------------------------

def test_f1_undefined_features_get_nan_and_a_reason_never_a_value():
    # constant series, a series below every guard (spectral entropy needs 8 points), constant quarterly
    for y, m in [(np.full(60, 5.0), 12), (synthetic(7, 12, 0), 12), (np.full(30, 2.0), 4)]:
        vals, flags = compute_features_for_series(y, m)
        assert all(math.isnan(vals[name]) for name in FEATURE_NAMES)
        assert all(flag.startswith("undefined:") for flag in flags.values()), flags


def test_f1_a_constant_rolling_window_makes_only_evolving_seasonality_undefined():
    y = synthetic(80, 12, 2)
    y[:40] = 7.0
    vals, flags = compute_features_for_series(y, 12)
    assert flags.pop("feature_quality_flag_evolving_seasonality") == "undefined:stl_constant_input"
    assert set(flags.values()) == {"ok"}
    assert math.isnan(vals["feature_evolving_seasonality"])


def test_f1_non_finite_input_is_an_error_not_interpolated():
    y = synthetic(60, 12, 3)
    y[10] = np.nan
    vals, flags = compute_features_for_series(y, 12)
    assert all(flag.startswith("error:ValueError") for flag in flags.values())
    assert all(math.isnan(vals[name]) for name in FEATURE_NAMES)


def test_f1_a_failing_feature_is_recorded_and_the_others_still_computed(monkeypatch):
    def boom(y, season_length):
        raise RuntimeError("boom")

    monkeypatch.setitem(compute_all.FEATURE_FUNCTIONS, "feature_arch_stat", boom)
    vals, flags = compute_features_for_series(synthetic(60, 12, 4), 12)
    assert flags["feature_quality_flag_arch_stat"] == "error:RuntimeError: boom"
    assert math.isnan(vals["feature_arch_stat"])
    assert sum(flag == "ok" for flag in flags.values()) == 5


@pytest.mark.parametrize("module", [mstl_features, nonlinear, arch, spectral, structural_breaks])
def test_feature_definitions_contain_no_local_recovery(module):
    """A feature either returns its value or raises; only compute_all records the outcome."""
    tree = ast.parse(inspect.getsource(module))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Try)]


# --- U1 / C3: only the history before the first test window is used --------------------------

def test_u1_values_after_the_first_cutoff_do_not_reach_any_feature():
    m, h, windows, n = 12, 18, 3, 130
    end = history_end(n, h, windows)
    y = synthetic(n, m, 7)
    changed_after = y.copy()
    changed_after[end:] = np.random.default_rng(1).normal(5000, 900, n - end)
    raw, flags = compute_feature_table(frame({"A": y}), m, h, windows)
    raw_after, flags_after = compute_feature_table(frame({"A": changed_after}), m, h, windows)
    pd.testing.assert_frame_equal(raw, raw_after)
    pd.testing.assert_frame_equal(flags, flags_after)
    assert raw.loc[0, "history_end_t"] == end == 76 and raw.loc[0, "series_length"] == n
    assert features_ok(flags).all()

    changed_last_history_point = y.copy()
    changed_last_history_point[end - 1] += 60
    raw_moved, _ = compute_feature_table(frame({"A": changed_last_history_point}), m, h, windows)
    assert not np.allclose(raw[FEATURE_NAMES].to_numpy(), raw_moved[FEATURE_NAMES].to_numpy())


# --- U5 / F2: STL without fallback -------------------------------------------------------------

def test_u5_stl_raises_on_too_short_or_constant_input():
    decompose_series(synthetic(24, 12, 0), 12)  # exactly two seasons
    with pytest.raises(DecompositionError, match="stl_too_short"):
        decompose_series(synthetic(23, 12, 0), 12)
    with pytest.raises(DecompositionError, match="stl_constant_input"):
        decompose_series(np.full(40, 3.0), 12)
    with pytest.raises(DecompositionError):
        component_targets(synthetic(7, 4, 0), 4)   # the engine's path: no fallback either


def test_u5_stl_components_add_up_to_the_series():
    for m, n in [(12, 120), (4, 40)]:
        y = synthetic(n, m, 3)
        dec = decompose_series(y, m)
        assert np.max(np.abs(dec["trend"] + dec["seasonal"] + dec["residual"] - y)) <= 1e-9
        t = component_targets(y, m)           # the targets the models are trained on
        np.testing.assert_array_equal(t["raw"], y)
        assert np.max(np.abs(t["trend"] + t["seasonal"] + t["residual"] - y)) <= 1e-9
        assert np.max(np.abs(t["nonseasonal"] + t["seasonal"] - y)) <= 1e-9
        assert all(len(v) == n for v in t.values())


def test_u5_stl_components_unchanged_when_test_data_is_perturbed():
    y = synthetic(96, 12, 4)
    g = pd.DataFrame({"ds": np.arange(96), "y": y})
    perturbed = g.copy()
    perturbed.loc[60:, "y"] += 1000.0
    train, test = split_fold(g, train_end_idx=60, h=18)
    train_p, test_p = split_fold(perturbed, train_end_idx=60, h=18)
    assert not test["y"].equals(test_p["y"])
    a = component_targets(train["y"].to_numpy(), 12)
    b = component_targets(train_p["y"].to_numpy(), 12)
    assert all(len(v) == 60 for v in a.values())   # fitted on the training slice only
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_stl_sn_continues_the_last_seasonal_cycle():
    seasonal = component_targets(synthetic(40, 4, 5), 4)["seasonal"]
    np.testing.assert_array_equal(seasonal_continuation(seasonal, 10, 4), np.tile(seasonal[-4:], 3)[:10])
    with pytest.raises(ValueError, match="shorter than one season"):
        seasonal_continuation(seasonal[:3], 10, 4)
