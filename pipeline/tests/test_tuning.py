"""Tuning on pre-test data and frozen configurations (protocol Section 4.3, D12, D13; test U12)."""
from __future__ import annotations

import json

import numpy as np
import optuna
import pandas as pd
import pytest
from utilsforecast.processing import backtest_splits

from src.forecast import models, spaces, tuning
from src.forecast.tuning import TuningError

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 20260521
SETTINGS = {"num_samples": 3, "early_stopping": {"patience": 1, "check_steps": 2}}


@pytest.fixture(autouse=True)
def cpu_and_short_training(monkeypatch):
    monkeypatch.setenv("RERUN_ACCELERATOR", "cpu")
    monkeypatch.setattr(spaces, "MAX_STEPS", [6, 10])  # test budget only; the protocol's values are tested elsewhere


def canonical(n_series: int, m: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_series):
        n = int(rng.integers(50, 80))
        t = np.arange(1, n + 1)
        y = 100 + 0.3 * t + 6 * np.sin(2 * np.pi * t / m + i) + rng.normal(0, 1.5, n)
        rows += [{"unique_id": f"S{i:03d}", "t": int(tt), "y": float(v)} for tt, v in zip(t, y)]
    return pd.DataFrame(rows)


def tuning_rows(data: pd.DataFrame, h: int, frequency: str) -> pd.DataFrame:
    lengths = data.groupby("unique_id")["t"].max()
    return pd.DataFrame({"frequency": frequency, "unique_id": lengths.index,
                         "tuning_validation_end_t": (lengths - 3 * h).to_numpy(),
                         "tuning_train_end_t": (lengths - 4 * h).to_numpy()})


# --- U12 ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", ["raw", "nonseasonal", "trend", "seasonal", "residual"])
def test_u12_tuning_data_end_at_the_first_cutoff_and_validation_is_the_last_h(target):
    m, h = 4, 8
    data = canonical(10, m, seed=1)
    rows = tuning_rows(data, h, "quarterly")
    frame = tuning.tuning_frame(data, rows, "quarterly", target, m)
    end = dict(zip(rows["unique_id"], rows["tuning_validation_end_t"]))
    assert frame.groupby("unique_id")["ds"].max().to_dict() == end

    changed = data.copy()
    after = changed["t"] > changed["unique_id"].map(end)
    changed.loc[after, "y"] += 10_000.0
    pd.testing.assert_frame_equal(frame, tuning.tuning_frame(changed, rows, "quarterly", target, m))

    (_, train, valid), = list(backtest_splits(frame, n_windows=1, h=h, id_col="unique_id", time_col="ds", freq=1))
    for uid, g in valid.groupby("unique_id"):
        assert g["ds"].tolist() == list(range(end[uid] - h + 1, end[uid] + 1))
    assert (train.groupby("unique_id")["ds"].max() == pd.Series(end) - h).all()


def test_tuning_frame_refuses_missing_series_or_short_history():
    data = canonical(4, 4, seed=2)
    rows = tuning_rows(data, 8, "quarterly")
    with pytest.raises(TuningError, match="missing"):
        tuning.tuning_frame(data[data["unique_id"] != "S000"], rows, "quarterly", "raw", 4)
    with pytest.raises(TuningError, match="no tuning series"):
        tuning.tuning_frame(data, rows, "monthly", "raw", 12)


def test_each_study_has_its_own_seed():
    seeds = {tuning.study_seed(SEED, mdl, f, tg) for mdl in ("Ridge", "NHITS") for f in ("monthly", "quarterly")
             for tg in ("raw", "trend")}
    assert len(seeds) == 8 and tuning.study_seed(SEED, "Ridge", "monthly", "raw") == tuning.study_seed(SEED, "Ridge", "monthly", "raw")


# --- studies run with the Auto classes ------------------------------------------------------------

def test_ml_study_freezes_the_sampled_parameters_and_is_reproducible(tmp_path):
    m, h = 4, 8
    data = canonical(12, m, seed=3)
    frame = tuning.tuning_frame(data, tuning_rows(data, h, "quarterly"), "quarterly", "raw", m)
    a = tuning.tune_study("RandomForest", "quarterly", "raw", frame, h, m, SETTINGS, SEED, tmp_path / "a.sqlite")
    b = tuning.tune_study("RandomForest", "quarterly", "raw", frame, h, m, SETTINGS, SEED, tmp_path / "b.sqlite")
    assert a["params"] == b["params"] and a["best_validation_mase"] == b["best_validation_mase"]
    assert set(a["params"]) == {"target_transform", "n_estimators", "max_depth", "min_samples_leaf", "max_features"}
    assert np.isfinite(a["best_validation_mase"])
    stored = optuna.load_study(study_name="RandomForest__quarterly__raw", storage=f"sqlite:///{tmp_path / 'a.sqlite'}")
    assert len(stored.trials) == 3 and min(t.value for t in stored.trials) == a["best_validation_mase"]
    forecasts = models.fit_predict_global("RandomForest", a["params"], data.rename(columns={"t": "ds"}), h, m, SEED)
    assert len(forecasts) == 12


def test_archive_keeps_every_trial_with_its_value_and_duration(tmp_path):
    import time

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0))

    def objective(trial):
        time.sleep(0.05)
        return trial.suggest_float("x", 0.0, 1.0)

    study.optimize(objective, n_trials=3)
    tuning.archive_study(study, tmp_path / "s.sqlite", "probe")
    stored = optuna.load_study(study_name="probe", storage=f"sqlite:///{tmp_path / 's.sqlite'}")
    for original, copy in zip(study.trials, stored.trials):
        assert copy.params == original.params and copy.value == original.value
        assert copy.datetime_start == original.datetime_start and copy.datetime_complete == original.datetime_complete
        assert copy.user_attrs["duration_seconds"] >= 0.05


def test_neural_study_freezes_parameters_and_trained_steps(tmp_path):
    m, h = 4, 8
    data = canonical(10, m, seed=4)
    frame = tuning.tuning_frame(data, tuning_rows(data, h, "quarterly"), "quarterly", "trend", m)
    entry = tuning.tune_study("NLinear", "quarterly", "trend", frame, h, m, SETTINGS, SEED, tmp_path / "n.sqlite")
    assert set(entry["params"]) == {"input_size", "scaler_type", "learning_rate", "max_steps", "batch_size"}
    assert 1 <= entry["trained_steps"] <= entry["params"]["max_steps"]
    forecasts = models.fit_predict_global("NLinear", entry["params"], data.rename(columns={"t": "ds"}), h, m, SEED,
                                          trained_steps=entry["trained_steps"])
    assert len(forecasts) == 10


# --- frozen-configuration file --------------------------------------------------------------------

def _entry():
    return {"model": "Ridge", "frequency": "monthly", "target": "raw", "params": {"alpha": 1.0, "target_transform": "none"},
            "best_validation_mase": 0.9}


def test_frozen_configs_round_trip_and_tamper_detection(tmp_path):
    path = tmp_path / "configs_frozen.json"
    digest = tuning.write_frozen_configs(path, {"Ridge|monthly|raw": _entry()}, SETTINGS, {"bundle_sha256": "x"})
    record = tuning.load_frozen_configs(path)
    assert record["configs_sha256"] == digest
    again = tuning.write_frozen_configs(tmp_path / "again.json", {"Ridge|monthly|raw": _entry()}, SETTINGS, {"bundle_sha256": "x"})
    assert again == digest  # the hash does not depend on the creation time

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["entries"]["Ridge|monthly|raw"]["params"]["alpha"] = 2.0
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(TuningError, match="does not match its hash"):
        tuning.load_frozen_configs(path)


def test_frozen_configs_refuse_inconsistent_keys_and_missing_steps(tmp_path):
    tuning.write_frozen_configs(tmp_path / "a.json", {"Ridge|quarterly|raw": _entry()}, SETTINGS, {})
    with pytest.raises(TuningError, match="inconsistent"):
        tuning.load_frozen_configs(tmp_path / "a.json")
    neural = {"model": "NHITS", "frequency": "monthly", "target": "raw", "params": {}, "best_validation_mase": 1.0}
    tuning.write_frozen_configs(tmp_path / "b.json", {"NHITS|monthly|raw": neural}, SETTINGS, {})
    with pytest.raises(TuningError, match="no trained steps"):
        tuning.load_frozen_configs(tmp_path / "b.json")
