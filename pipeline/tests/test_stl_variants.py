"""STL variants of sensitivity analysis S1 (change log v2.2): log STL and periodic STL, STL-SN only.

The run's STL (the default) must be untouched: same targets and same forecasts, bit for bit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.validation import make_cutoffs
from src.features.errors import DecompositionError
from src.features.mstl_features import decompose_series
from src.forecast import engine, models
from src.forecast.engine import Provenance
from src.forecast.registry import TARGETS
from src.forecast.targets import component_targets, seasonal_continuation, stl_sn_forecast
from src.forecast.tuning import tuning_frame

M, H = 4, 8
L = 3 * M + H
PROV = Provenance("c" * 40, "e" * 64, "d" * 64, "b" * 64, "f" * 64)
RIDGE = {f"Ridge|quarterly|{t}": {"model": "Ridge", "frequency": "quarterly", "target": t,
                                  "params": {"alpha": 1.0, "target_transform": "none"}} for t in TARGETS}


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("RERUN_ACCELERATOR", "cpu")


def multiplicative(n: int, seed: int) -> np.ndarray:
    """A positive series whose seasonal swing grows with its level."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    level = 100 * np.exp(0.01 * t)
    return level * (1 + 0.2 * np.sin(2 * np.pi * t / M)) * np.exp(rng.normal(0, 0.02, n))


@pytest.fixture(scope="module")
def data():
    rows = []
    for i in range(5):
        y = multiplicative(L + 3 * H + 3 * i, seed=i)
        rows += [{"unique_id": f"M4_Quarterly_Q{i}", "source_dataset": "M4_Quarterly", "t": t + 1, "ds": t + 1, "y": float(v)}
                 for t, v in enumerate(y)]
    df = pd.DataFrame(rows)
    return df, make_cutoffs(df, H, 3, H, min_history=L)


def _task(strategy, variant=None):
    t = {"feature_name": "feature_arch_stat", "frequency": "quarterly", "strategy": strategy, "model": "Ridge",
         "scope": "cohort", "seed": 7}
    return t if variant is None else {**t, "stl_variant": variant}


def _run(data, strategy, variant=None):
    df, cutoffs = data
    buckets = {uid: "Low" for uid in df["unique_id"].unique()}
    return engine.evaluate_global_task(_task(strategy, variant), df, cutoffs, RIDGE, buckets, H, M, PROV)


# --- targets --------------------------------------------------------------------------------------

def test_default_targets_are_the_runs_stl():
    y = multiplicative(40, 1)
    dec = decompose_series(y, M)
    t = component_targets(y, M)
    assert t == {**t} and np.array_equal(t["nonseasonal"], dec["trend"] + dec["residual"])
    for name in ("trend", "seasonal", "residual"):
        assert np.array_equal(t[name], dec[name])
    assert all(np.array_equal(t[k], v) for k, v in component_targets(y, M, "default").items())


def test_log_stl_decomposes_the_log_and_reconstructs_the_series():
    y = multiplicative(40, 2)
    t = component_targets(y, M, "log")
    assert np.array_equal(t["raw"], y)
    np.testing.assert_allclose(np.exp(t["trend"] + t["seasonal"] + t["residual"]), y, rtol=1e-10)
    np.testing.assert_allclose(t["nonseasonal"], decompose_series(np.log(y), M)["trend"] + decompose_series(np.log(y), M)["residual"])
    with pytest.raises(DecompositionError, match="nonpositive"):
        component_targets(np.r_[y[:-1], 0.0], M, "log")


def test_periodic_stl_repeats_one_seasonal_pattern():
    y = multiplicative(40, 3)
    s = component_targets(y, M, "periodic")["seasonal"]
    np.testing.assert_allclose(s[M:], s[:-M], atol=1e-8)
    assert not np.allclose(component_targets(y, M)["seasonal"][M:], component_targets(y, M)["seasonal"][:-M], atol=1e-8)


def test_unknown_variant_is_refused():
    with pytest.raises(ValueError, match="unknown STL variant"):
        component_targets(multiplicative(40, 4), M, "x13")


def test_stl_sn_forecast_recomposition():
    f, s = np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4])
    expected = f + seasonal_continuation(s, 5, M)
    assert np.array_equal(stl_sn_forecast(f, s, 5, M), expected)
    np.testing.assert_allclose(stl_sn_forecast(f, s, 5, M, "periodic"), expected)
    np.testing.assert_allclose(stl_sn_forecast(f, s, 5, M, "log"), np.exp(expected))


# --- engine ------------------------------------------------------------------------------------------

def test_default_global_forecasts_are_bit_identical_to_the_runs_recomposition(data):
    df, cutoffs = data
    rows = _run(data, "stl_sn")
    assert "stl_variant" not in rows.columns
    w0 = cutoffs[cutoffs["window"] == 0].set_index("unique_id")["train_end_idx"]
    parts, comps = [], {}
    for uid, g in df.sort_values("t").groupby("unique_id"):
        comps[uid] = component_targets(g["y"].to_numpy()[: w0[uid]], M)
        parts.append(pd.DataFrame({"unique_id": uid, "ds": np.arange(1, w0[uid] + 1), "y": comps[uid]["nonseasonal"]}))
    fc = models.fit_predict_global("Ridge", RIDGE["Ridge|quarterly|nonseasonal"]["params"], pd.concat(parts), H, M, 7)
    for uid in comps:
        expected = fc[uid].yhat + seasonal_continuation(comps[uid]["seasonal"], H, M)   # the run's expression
        np.testing.assert_array_equal(rows.query("window == 0 and unique_id == @uid")["yhat"].iloc[0], expected)


@pytest.mark.parametrize("variant", ["log", "periodic"])
def test_variant_global_rows_carry_the_variant_and_recompose(data, variant):
    df, cutoffs = data
    rows = _run(data, "stl_sn", variant)
    assert (rows["stl_variant"] == variant).all() and len(rows) == 15 and (rows["status"] == "trained").all()
    assert np.isfinite(rows["relnaive_capped"]).all()
    w0 = cutoffs[cutoffs["window"] == 0].set_index("unique_id")["train_end_idx"]
    parts, comps = [], {}
    for uid, g in df.sort_values("t").groupby("unique_id"):
        comps[uid] = component_targets(g["y"].to_numpy()[: w0[uid]], M, variant)
        parts.append(pd.DataFrame({"unique_id": uid, "ds": np.arange(1, w0[uid] + 1), "y": comps[uid]["nonseasonal"]}))
    fc = models.fit_predict_global("Ridge", RIDGE["Ridge|quarterly|nonseasonal"]["params"], pd.concat(parts), H, M, 7)
    for uid in comps:
        expected = stl_sn_forecast(fc[uid].yhat, comps[uid]["seasonal"], H, M, variant)
        np.testing.assert_allclose(rows.query("window == 0 and unique_id == @uid")["yhat"].iloc[0], expected, rtol=1e-12)


@pytest.mark.parametrize("strategy", ["direct", "stl_ac"])
def test_variants_are_refused_outside_stl_sn(data, strategy):
    with pytest.raises(ValueError, match="STL-SN only"):
        _run(data, strategy, "log")
    df, cutoffs = data
    one = df[df["unique_id"] == "M4_Quarterly_Q0"]
    with pytest.raises(ValueError, match="STL-SN only"):
        engine.evaluate_statistical_series("ETS", "quarterly", strategy, one, cutoffs, H, M, PROV, stl_variant="periodic")


def test_statistical_variant_rows(data):
    df, cutoffs = data
    one = df[df["unique_id"] == "M4_Quarterly_Q1"]
    base = engine.evaluate_statistical_series("ETS", "quarterly", "stl_sn", one, cutoffs, H, M, PROV)
    log = engine.evaluate_statistical_series("ETS", "quarterly", "stl_sn", one, cutoffs, H, M, PROV, stl_variant="log")
    assert "stl_variant" not in base.columns and (log["stl_variant"] == "log").all()
    assert (log["status"] == "trained").all() and len(log) == 3
    assert not np.allclose(np.stack(base["yhat"]), np.stack(log["yhat"]))


def test_tuning_frame_uses_the_variants_target(data):
    df, cutoffs = data
    w0 = cutoffs[cutoffs["window"] == 0]
    tuning_set = pd.DataFrame({"unique_id": w0["unique_id"], "frequency": "quarterly",
                               "tuning_validation_end_t": w0["train_end_idx"]})
    base = tuning_frame(df, tuning_set, "quarterly", "nonseasonal", M)
    log = tuning_frame(df, tuning_set, "quarterly", "nonseasonal", M, stl_variant="log")
    assert len(base) == len(log) and (log["y"] < 10).all() and (base["y"] > 50).all()   # log scale vs raw scale
