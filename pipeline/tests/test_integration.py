"""Integration tests I1 (full chain) and I4 (resume) with real model fits on CPU (protocol Section 7.2).

Run with ``python -m pytest tests -m integration``. The data are the synthetic frozen copy of
helpers_run (one feature, both frequencies). I1 narrows two values of the neural search space
(max_steps 20, standard scaler) so that the chain finishes in minutes; the protocol's space is
unchanged and is checked by test_forecast_models.
"""
from __future__ import annotations

import copy
import shutil

import optuna
import pandas as pd
import pytest
import yaml

from helpers_run import COMMIT, CONFIG, complete_studies, synthetic_frozen_copy
from src import cli
from src.data import frozen
from src.forecast import engine, spaces
from src.forecast.registry import FAMILY
from src.stages import io
from src.stages.evaluate import run_evaluate
from src.stages.io import read_json
from src.stages.prepare import run_prepare
from src.stages.tasks import evaluation_tasks, tuning_tasks
from src.stages.tune import run_freeze

pytestmark = pytest.mark.integration

ONE_FEATURE = {**copy.deepcopy(CONFIG), "features": ["feature_evolving_seasonality"], "seed_check_seeds": [],
               "tuning": {"num_samples": 1, "early_stopping": {"patience": 1, "check_steps": 5}}}


@pytest.fixture
def cpu_run(monkeypatch):
    monkeypatch.setenv("RERUN_CODE_COMMIT", COMMIT)
    monkeypatch.setenv("RERUN_ACCELERATOR", "cpu")
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def test_i1_the_full_chain_runs_through_the_cli(tmp_path, monkeypatch, cpu_run):
    data_dir, manifest_path = synthetic_frozen_copy(tmp_path / "data")
    monkeypatch.setattr(frozen, "DEFAULT_MANIFEST", manifest_path)
    monkeypatch.setattr(spaces, "MAX_STEPS", [20])
    monkeypatch.setattr(spaces, "SCALER", ["standard"])
    config_path, run, out = tmp_path / "config.yaml", tmp_path / "run", tmp_path / "analysis"
    config_path.write_text(yaml.safe_dump(ONE_FEATURE), encoding="utf-8")
    common = ["--config", str(config_path), "--run", str(run)]

    assert cli.main(["freeze-data", "verify", "--data-dir", str(data_dir), "--manifest", str(manifest_path)]) == 0
    assert cli.main(["prepare", *common, "--data-dir", str(data_dir), "--jobs", "2"]) == 0
    for shard in sorted({t.shard for t in tuning_tasks()}):
        assert cli.main(["tune", *common, "--data-dir", str(data_dir), "--shard", shard]) == 0
    assert cli.main(["freeze", *common]) == 0
    prepare = run / "prepare"
    tasks = evaluation_tasks(pd.read_parquet(prepare / "bucket_summary.parquet"), pd.read_parquet(prepare / "samples.parquet"),
                             ONE_FEATURE["random_seed"], [])
    for shard in sorted({t.shard for t in tasks}):
        assert cli.main(["evaluate", *common, "--data-dir", str(data_dir), "--shard", shard]) == 0
    assert cli.main(["merge", "--run", str(run)]) == 0
    gates_exit = cli.main(["gates", "--run", str(run)])
    report = read_json(run / "merged" / "gates.json")
    assert gates_exit == 0, {k: v for k, v in report["gates"].items() if not v["passed"]}
    assert cli.main(["analyse", "--run", str(run), "--out", str(out)]) == 0

    configs = read_json(run / "configs_frozen.json")
    assert len(configs["entries"]) == 90
    assert all(e["trained_steps"] >= 1 for e in configs["entries"].values() if FAMILY[e["model"]] != "ml")
    merge = read_json(run / "merged" / "merge.json")
    assert merge["grid"]["completed"] == merge["grid"]["expected"] == len(tasks)
    rows = pd.read_parquet(run / "merged" / "rows.parquet")
    assert set(rows["model"]) == set(FAMILY) and (rows["backend_version"] != "test").all()
    assert set(read_json(out / "analysis.json")["files"]) >= {"tables/h1.csv", "figures/cd_cohort_all.png"}


def _sampled_params(task):
    trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=len(task.task_id))).ask()
    if FAMILY[task.model] == "ml":
        return spaces.sample_ml(task.model, trial)
    return spaces.sample_neural(task.model, trial, 12 if task.frequency == "monthly" else 4)


def test_i4_a_shard_killed_mid_run_resumes_to_identical_outputs(tmp_path, monkeypatch, cpu_run):
    data_dir, manifest_path = synthetic_frozen_copy(tmp_path / "data")
    config, shard = ONE_FEATURE, "ml-quarterly"
    killed, clean = tmp_path / "killed", tmp_path / "clean"
    bundle = run_prepare(config, killed, data_dir, COMMIT, manifest_path=manifest_path)
    complete_studies(killed, io.stage_provenance(bundle, manifest_path), params_for=_sampled_params)
    run_freeze(killed, config)
    shutil.copytree(killed, clean)

    run_evaluate(clean, config, data_dir, shard, manifest_path)

    real, calls = engine.evaluate_global_task, []

    def killed_after_ten(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 11:
            raise KeyboardInterrupt  # what a kill does: not an Exception, so not retried
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "evaluate_global_task", killed_after_ten)
    with pytest.raises(KeyboardInterrupt):
        run_evaluate(killed, config, data_dir, shard, manifest_path)
    tasks_dir = killed / "evaluate" / shard / "tasks"
    records = sorted(tasks_dir.glob("*.done.json"))
    assert len(records) == 10
    records[-1].unlink()  # killed after the output was written but before its record
    calls.clear()
    monkeypatch.setattr(engine, "evaluate_global_task", lambda *a, **k: (calls.append(a[0]), real(*a, **k))[1])
    run_evaluate(killed, config, data_dir, shard, manifest_path)
    assert len(calls) == len(list((clean / "evaluate" / shard / "tasks").glob("*.done.json"))) - 9  # nothing redone

    timing = ["fit_seconds", "predict_seconds"]
    clean_dir = clean / "evaluate" / shard / "tasks"
    assert sorted(p.name for p in tasks_dir.glob("*.done.json")) == sorted(p.name for p in clean_dir.glob("*.done.json"))
    for record in sorted(clean_dir.glob("*.done.json")):
        name = read_json(record)["file"]
        a = pd.read_parquet(tasks_dir / name).drop(columns=timing)
        b = pd.read_parquet(clean_dir / name).drop(columns=timing)
        pd.testing.assert_frame_equal(a, b)
