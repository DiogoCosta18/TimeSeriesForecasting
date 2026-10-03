"""Evaluation engine (protocol Sections 4.1, 4.4, 4.5, 5.1; D16, D18, D19; tests U13, U16, U17)."""
from __future__ import annotations

import ast
import inspect
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from src.data.validation import make_cutoffs
from src.forecast import engine, models
from src.forecast.engine import Provenance
from src.forecast.metrics import pocid, row_metrics
from src.forecast.registry import TARGETS
from src.forecast.targets import component_targets, seasonal_continuation

M, H = 4, 8
L = 3 * M + H
PROV = Provenance("c" * 40, "e" * 64, "d" * 64, "b" * 64, "f" * 64)
RIDGE = {f"Ridge|quarterly|{t}": {"model": "Ridge", "frequency": "quarterly", "target": t,
                                  "params": {"alpha": 1.0, "target_transform": "none"}} for t in TARGETS}


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("RERUN_ACCELERATOR", "cpu")


def canonical(n_series: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_series):
        n = L + 3 * H + int(rng.integers(0, 20))
        t = np.arange(1, n + 1)
        y = 100 + 0.4 * t + 7 * np.sin(2 * np.pi * t / M + i) + rng.normal(0, 1.5, n)
        rows += [{"unique_id": f"M4_Quarterly_Q{i}", "source_dataset": "M4_Quarterly", "t": int(tt), "ds": int(tt), "y": float(v)}
                 for tt, v in zip(t, y)]
    return pd.DataFrame(rows)


def task(strategy: str, model: str = "Ridge") -> dict:
    return {"feature_name": "feature_arch_stat", "frequency": "quarterly", "strategy": strategy, "model": model,
            "scope": "cohort", "seed": 7}


@pytest.fixture(scope="module")
def data():
    df = canonical(6, seed=1)
    return df, make_cutoffs(df, H, 3, H, min_history=L)


def run(data, strategy, entries=RIDGE, model="Ridge"):
    df, cutoffs = data
    buckets = {uid: "Low" for uid in df["unique_id"].unique()}
    return engine.evaluate_global_task(task(strategy, model), df, cutoffs, entries, buckets, H, M, PROV)


# --- windows, rows and provenance ---------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["direct", "stl_sn", "stl_ac"])
def test_global_rows_follow_the_windows_and_carry_full_provenance(data, strategy):
    df, _ = data
    rows = run(data, strategy)
    assert len(rows) == 6 * 3 and (rows["status"] == "trained").all()
    lengths = df.groupby("unique_id")["t"].max()
    for _, r in rows.iterrows():
        n = lengths[r["unique_id"]]
        assert r["cutoff_t"] == n - (3 - r["window"]) * H
        y = df[df["unique_id"] == r["unique_id"]].sort_values("t")["y"].to_numpy()
        assert r["y_test"] == list(y[r["cutoff_t"]: r["cutoff_t"] + H])
        assert len(r["yhat"]) == len(r["yhat_naive"]) == H
    for field, value in asdict(PROV).items():
        assert (rows[field] == value).all()
    for col in ["relnaive_capped", "relnaive", "mase", "smape", "mae", "pocid", "fit_seconds", "predict_seconds"]:
        assert rows[col].notna().all()
    assert (rows["relnaive_capped"] <= 10).all() and (rows["bucket"] == "Low").all() and (rows["seed"] == 7).all()
    n_components = {"direct": 1, "stl_sn": 1, "stl_ac": 3}[strategy]
    assert rows["components"].map(lambda c: len(json.loads(c))).eq(n_components).all()


def test_stl_strategies_recompose_exactly_as_the_protocol_defines(data):
    df, cutoffs = data
    sn, ac = run(data, "stl_sn"), run(data, "stl_ac")
    w0 = cutoffs[cutoffs["window"] == 0].set_index("unique_id")["train_end_idx"]
    parts = {t: [] for t in ("nonseasonal", "trend", "seasonal", "residual")}
    comps = {}
    for uid, g in df.sort_values("t").groupby("unique_id"):
        comps[uid] = component_targets(g["y"].to_numpy()[: w0[uid]], M)
        for t in parts:
            parts[t].append(pd.DataFrame({"unique_id": uid, "ds": np.arange(1, w0[uid] + 1), "y": comps[uid][t]}))
    fc = {t: models.fit_predict_global("Ridge", RIDGE[f"Ridge|quarterly|{t}"]["params"], pd.concat(p), H, M, 7)
          for t, p in parts.items()}
    for uid in comps:
        expected_sn = fc["nonseasonal"][uid].yhat + seasonal_continuation(comps[uid]["seasonal"], H, M)
        expected_ac = fc["trend"][uid].yhat + fc["seasonal"][uid].yhat + fc["residual"][uid].yhat
        np.testing.assert_array_equal(sn.query("window == 0 and unique_id == @uid")["yhat"].iloc[0], expected_sn)
        np.testing.assert_array_equal(ac.query("window == 0 and unique_id == @uid")["yhat"].iloc[0], expected_ac)


def test_u13_stl_ac_rows_store_three_component_provenances(data):
    rows = run(data, "stl_ac")
    comps = json.loads(rows["components"].iloc[0])
    assert [c["target"] for c in comps] == ["trend", "seasonal", "residual"]
    assert [c["config_key"] for c in comps] == ["Ridge|quarterly|trend", "Ridge|quarterly|seasonal", "Ridge|quarterly|residual"]
    assert all(c["backend"] == "mlforecast" and len(c["forecast_hash"]) == 64 for c in comps)


def test_neural_global_task_runs_from_its_frozen_entry(data):
    entries = {"NLinear|quarterly|raw": {"model": "NLinear", "frequency": "quarterly", "target": "raw", "trained_steps": 3,
                                        "params": {"input_size": 8, "scaler_type": "standard", "learning_rate": 1e-3,
                                                   "max_steps": 500, "batch_size": 32}}}
    rows = run(data, "direct", entries, model="NLinear")
    assert len(rows) == 18 and (rows["backend"] == "neuralforecast").all() and (rows["device"] == "cpu").all()


# --- U17: failures ----------------------------------------------------------------------------------

def test_u17_a_global_exception_fails_the_task_with_no_rows(data, monkeypatch):
    calls = {"n": 0}
    original = models.fit_predict_global

    def fail_second_window(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("backend crashed")
        return original(*args, **kwargs)

    monkeypatch.setattr(models, "fit_predict_global", fail_second_window)
    with pytest.raises(RuntimeError, match="backend crashed"):
        run(data, "direct")


def test_u17_and_u13_a_statistical_failure_is_a_failed_row_and_the_series_continues(data, monkeypatch):
    df, cutoffs = data
    one = df[df["unique_id"] == "M4_Quarterly_Q0"]
    original = models.forecast_statistical
    seen = {"n": 0}

    def fail_seasonal_component_once(model, y, h, m, timings=None):
        seen["n"] += 1
        if seen["n"] == 2:  # window 0, second component (seasonal)
            raise ValueError("no convergence")
        return original(model, y, h, m, timings=timings)

    monkeypatch.setattr(models, "forecast_statistical", fail_seasonal_component_once)
    rows = engine.evaluate_statistical_series("ETS", "quarterly", "stl_ac", one, cutoffs, H, M, PROV)
    assert rows["status"].tolist() == ["failed", "trained", "trained"]
    failed = rows.iloc[0]
    assert failed["failure"] == "ValueError: no convergence" and failed["yhat"] is None and np.isnan(failed["relnaive_capped"])
    assert len(json.loads(rows.iloc[1]["components"])) == 3


# --- U16: statistical forecasts are pool-independent ---------------------------------------------------

def test_u16_statistical_rows_are_identical_for_every_feature_sample(data):
    df, cutoffs = data
    one = df[df["unique_id"] == "M4_Quarterly_Q1"]
    for_sample_a = engine.evaluate_statistical_series("ARIMA", "quarterly", "direct", one, cutoffs, H, M, PROV)
    for_sample_b = engine.evaluate_statistical_series("ARIMA", "quarterly", "direct", one, cutoffs, H, M, PROV)
    assert for_sample_a["forecast_hash"].tolist() == for_sample_b["forecast_hash"].tolist()
    assert for_sample_a["feature_name"].isna().all() and (for_sample_a["scope"] == "cohort").all()
    assert "feature" not in " ".join(inspect.signature(engine.evaluate_statistical_series).parameters)


def test_statistical_models_are_not_run_as_global_tasks(data):
    with pytest.raises(ValueError, match="per series"):
        run(data, "direct", RIDGE, model="ARIMA")


# --- metrics and structure -------------------------------------------------------------------------------

def test_pocid_follows_the_papers_equation():
    # anchor 10: actual changes +2, -1, 0, +3; forecast changes +1, +1, +1, +2 -> agree, disagree, zero, agree
    assert pocid(10.0, [12.0, 11.0, 11.0, 14.0], [11.0, 12.0, 13.0, 15.0]) == 0.5
    keys = row_metrics(np.arange(20.0), [1.0, 2.0], [1.0, 2.0], [1.0, 1.0], 4).keys()
    assert list(keys) == ["relnaive_capped", "relnaive", "mase", "smape", "mae", "pocid"]


def test_engine_has_exactly_one_failure_handler_the_statistical_failed_row():
    tree = ast.parse(inspect.getsource(engine))
    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
    assert len(handlers) == 1
    owner = [f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and handlers[0] in list(ast.walk(f))]
    assert owner == ["evaluate_statistical_series"]
