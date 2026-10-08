"""Sensitivity analysis S1 (change log v2.2) on a synthetic run that went through merge and gates.

The fits and the studies are stand-ins (the stage's bookkeeping, pairing, checks and statistics
are real); the STL variants themselves are covered by test_stl_variants.
"""
from __future__ import annotations

import copy
import json
import shutil

import optuna
import pandas as pd
import pytest

from helpers_run import COMMIT, CONFIG, evaluated_run, fake_fits
from src.forecast import tuning
from src.stages import sensitivity as s1
from src.stages.io import StageError, read_json
from src.stages.merge import run_merge
from src.validation.gates import run_gates

SMALL = {**copy.deepcopy(CONFIG), "features": ["feature_evolving_seasonality"], "seed_check_seeds": [8]}


def fake_study(model, frequency, target, frame, h, m, settings, seed):
    study = optuna.create_study(direction="minimize")
    study.add_trial(optuna.trial.create_trial(params={}, distributions={}, value=1.0))
    entry = {"model": model, "frequency": frequency, "target": target, "params": {}, "best_validation_mase": 1.0,
             "n_tuning_series": int(frame["unique_id"].nunique()), "trained_steps": 1}
    return entry, tuning.archived_trials(study, f"{model}|{frequency}|{target}")


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    run, data_dir, manifest = evaluated_run(tmp_path_factory.mktemp("s1"), SMALL)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RERUN_CODE_COMMIT", COMMIT)
        run_merge(run, manifest)
        assert run_gates(run, manifest)["passed"]
    return run, data_dir, manifest


@pytest.fixture
def run(merged, tmp_path, monkeypatch):
    source, data_dir, manifest = merged
    shutil.copytree(source, tmp_path / "run")
    monkeypatch.setenv("RERUN_CODE_COMMIT", COMMIT)
    fake_fits(monkeypatch)                                   # inherited by forked workers
    monkeypatch.setattr(tuning, "run_study", fake_study)
    return tmp_path / "run", data_dir, manifest


def _complete(run_dir, data_dir, manifest):
    for group in ("cpu", "gpu"):
        s1.run_tune(run_dir, SMALL, data_dir, group, manifest)
    s1.run_freeze(run_dir)
    s1.run_evaluate(run_dir, SMALL, data_dir, "cpu", manifest, workers=2)
    s1.run_evaluate(run_dir, SMALL, data_dir, "gpu", manifest)


def test_task_grid():
    summary = pd.DataFrame({"feature_name": ["feature_evolving_seasonality"] * 2, "frequency": ["monthly", "quarterly"]})
    samples = pd.DataFrame({"frequency": ["monthly"] * 300 + ["quarterly"] * 10, "unique_id": [f"s{i}" for i in range(310)]})
    tasks = s1.eval_tasks(summary, samples)
    glob = [t for t in tasks if t["kind"] == "global"]
    stat = [t for t in tasks if t["kind"] == "statistical"]
    assert len(glob) == 2 * 2 * 1 * 3 and {t["model"] for t in glob} == {"XGBoost", "NHITS", "PatchTST"}
    assert len(stat) == (2 + 1) * 2 and {t["group"] for t in stat} == {"cpu"}        # 300 -> 2 chunks of 250, 10 -> 1
    assert {t["group"] for t in glob if t["model"] != "XGBoost"} == {"gpu"}
    assert len(s1.tune_tasks()) == 12 and len({t["task_id"] for t in tasks}) == len(tasks)


def test_s1_end_to_end(run):
    run_dir, data_dir, manifest = run
    _complete(run_dir, data_dir, manifest)
    frozen = s1.load_frozen(run_dir)
    assert len(frozen["entries"]) == 12 and {e["stl_variant"] for e in frozen["entries"].values()} == {"log", "periodic"}
    result = s1.run_analyse(run_dir, SMALL, manifest)
    out = run_dir / "sensitivity_s1" / "analysis"
    checks = read_json(out / "checks.json")
    assert checks["passed"] and all(c["passed"] for c in checks["checks"].values())
    assert checks["checks"]["G6_pairing"]["unpaired_with_direct"] == 0 and checks["checks"]["G6_pairing"]["s1_instances"] > 0
    tests = pd.read_csv(out / "tables" / "s1_tests.csv")
    assert len(tests) == 8 and set(tests["n_tests_in_family"]) == {8}
    assert set(zip(tests["model"], tests["variant"])) == {(m, v) for m in s1.S1_MODELS.values() for v in s1.VARIANTS}
    default = pd.read_csv(out / "tables" / "s1_default.csv")
    assert sorted(default["model"]) == sorted(s1.S1_MODELS.values())
    terciles = pd.read_csv(out / "tables" / "s1_evolving_terciles.csv")
    assert set(terciles["variant"]) == {"log", "periodic", "default"}
    summary = read_json(out / "s1.json")
    assert summary["decision"] == result["decision"] and "not a confirmatory hypothesis" in summary["status"]
    with pytest.raises(StageError, match="never overwritten"):
        s1.run_analyse(run_dir, SMALL, manifest)


def test_analysis_refuses_an_incomplete_s1(run):
    run_dir, data_dir, manifest = run
    _complete(run_dir, data_dir, manifest)
    record = next((run_dir / "sensitivity_s1" / "evaluate" / "tasks").glob("s1stat*.done.json"))
    record.unlink()
    with pytest.raises(StageError, match="S1 checks failed"):
        s1.run_analyse(run_dir, SMALL, manifest)
    checks = json.loads((run_dir / "sensitivity_s1" / "analysis" / "checks.json").read_text())
    assert not checks["checks"]["G5_complete"]["passed"] and len(checks["checks"]["G5_complete"]["missing"]) == 1


def test_freeze_and_evaluate_refuse_out_of_order(run):
    run_dir, data_dir, manifest = run
    s1.run_tune(run_dir, SMALL, data_dir, "cpu", manifest)
    with pytest.raises(StageError, match="has not completed"):
        s1.run_freeze(run_dir)
    with pytest.raises(FileNotFoundError):
        s1.run_evaluate(run_dir, SMALL, data_dir, "cpu", manifest)
    with pytest.raises(StageError, match="--workers is for the cpu group"):
        s1.run_evaluate(run_dir, SMALL, data_dir, "gpu", manifest, workers=2)
