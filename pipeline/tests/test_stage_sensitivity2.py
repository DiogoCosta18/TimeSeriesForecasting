"""Sensitivity analysis S2 (change log v2.4) on a synthetic run that went through merge and gates.

The fits and the studies are stand-ins, as in test_stage_sensitivity (the stage's bookkeeping,
pairing, checks and statistics are real); the variants themselves are covered by test_stl_variants.
"""
from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest

from helpers_run import COMMIT, evaluated_run, fake_fits
from src.forecast import tuning
from src.stages import sensitivity as s1
from src.stages import sensitivity2 as s2
from src.stages.io import StageError, read_json
from src.stages.merge import run_merge
from src.validation.gates import run_gates
from test_stage_sensitivity import SMALL, fake_study


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    run, data_dir, manifest = evaluated_run(tmp_path_factory.mktemp("s2"), SMALL)
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
    fake_fits(monkeypatch)
    monkeypatch.setattr(tuning, "run_study", fake_study)
    return tmp_path / "run", data_dir, manifest


def _complete(run_dir, data_dir, manifest):
    for group in ("cpu", "gpu"):
        s1.run_tune(run_dir, SMALL, data_dir, group, manifest, study=s2.S2)
    s1.run_freeze(run_dir, study=s2.S2)
    s1.run_evaluate(run_dir, SMALL, data_dir, "cpu", manifest, workers=2, study=s2.S2)
    s1.run_evaluate(run_dir, SMALL, data_dir, "gpu", manifest, study=s2.S2)


def test_task_grid():
    summary = pd.DataFrame({"feature_name": ["feature_evolving_seasonality"] * 2, "frequency": ["monthly", "quarterly"]})
    samples = pd.DataFrame({"frequency": ["monthly"] * 300 + ["quarterly"] * 10, "unique_id": [f"s{i}" for i in range(310)]})
    tasks = s2.eval_tasks(summary, samples)
    glob = [t for t in tasks if t["kind"] == "global"]
    stat = [t for t in tasks if t["kind"] == "statistical"]
    assert len(glob) == 2 * 2 * 1 * 3 and {t["variant"] for t in glob} == {"deg0", "last3"}
    assert len(stat) == (2 + 1) * 4 and {(t["model"], t["variant"]) for t in stat} == set(s2.STATISTICAL)
    assert len(s2.tune_tasks()) == 6 and {t["variant"] for t in s2.tune_tasks()} == {"deg0"}
    assert len({t["task_id"] for t in tasks}) == len(tasks) and all(t["task_id"].startswith("s2") for t in tasks)
    assert len(s2.CONTRASTS) == 10


def test_s1_is_unchanged_by_the_study_parameter():
    summary = pd.DataFrame({"feature_name": ["feature_evolving_seasonality"], "frequency": ["monthly"]})
    samples = pd.DataFrame({"frequency": ["monthly"] * 10, "unique_id": [f"s{i}" for i in range(10)]})
    assert s1.S1.root == "sensitivity_s1" and s1.S1.tune_tasks() == s1.tune_tasks()
    assert s1.S1.eval_tasks(summary, samples) == s1.eval_tasks(summary, samples) and s1.S1.run_entries is None


def test_s2_end_to_end(run):
    run_dir, data_dir, manifest = run
    _complete(run_dir, data_dir, manifest)
    frozen = s1.load_frozen(run_dir, s2.S2)
    entries = frozen["entries"]
    assert len(entries) == 12 and {e["stl_variant"] for e in entries.values()} == {"deg0", "last3"}
    runs = tuning.load_frozen_configs(run_dir / "configs_frozen.json")
    for key, e in entries.items():
        if e["stl_variant"] == "last3":               # the run's configuration of the non-seasonal target
            model, frequency = key.split("|")[:2]
            original = runs["entries"][f"{model}|{frequency}|nonseasonal"]
            assert e["params"] == original["params"] and e["taken_from_run_configs"] == runs["configs_sha256"]
    assert not (run_dir / "sensitivity_s1").exists()  # S2 writes in its own directory only
    result = s2.run_analyse(run_dir, SMALL, manifest)
    out = run_dir / "sensitivity_s2" / "analysis"
    checks = read_json(out / "checks.json")
    assert checks["passed"] and checks["checks"]["G6_pairing"]["unpaired_with_run_stl_sn"] == 0
    for name in ("s2_vs_direct", "s2_vs_run_stl_sn"):
        t = pd.read_csv(out / "tables" / f"{name}.csv")
        assert len(t) == 10 and set(t["n_tests_in_family"]) == {10}
        assert set(zip(t["model"], t["variant"])) == set(s2.CONTRASTS)
    same = pd.read_csv(out / "tables" / "s2_last3_identical.csv")
    assert set(same["model"]) == {"ETS", "XGBoost", "NHITS", "PatchTST"} and (same["identical_share"] == 1).all()
    summary = read_json(out / "s2.json")
    assert summary["reading"] == json.loads(json.dumps(result["reading"])) and "not a confirmatory" in summary["status"]
    assert set(summary["reading"]["stlf"]) == {"ETS", "ARIMA"}
    with pytest.raises(StageError, match="never overwritten"):
        s2.run_analyse(run_dir, SMALL, manifest)


def test_analysis_refuses_an_incomplete_s2(run):
    run_dir, data_dir, manifest = run
    _complete(run_dir, data_dir, manifest)
    next((run_dir / "sensitivity_s2" / "evaluate" / "tasks").glob("s2stat__ARIMA*.done.json")).unlink()
    with pytest.raises(StageError, match="S2 checks failed"):
        s2.run_analyse(run_dir, SMALL, manifest)
