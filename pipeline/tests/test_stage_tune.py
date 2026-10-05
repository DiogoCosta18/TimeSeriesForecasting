"""Stage R2 tune: worker processes, retry-once and failure listing (D16, Section 9.4).

The worker test tunes a real ML shard (one trial per study) on the synthetic frozen copy;
the bookkeeping tests replace the study by a stand-in that counts calls.
"""
from __future__ import annotations

import shutil

import optuna
import pytest

from helpers_run import COMMIT, CONFIG, synthetic_frozen_copy
from src.forecast import tuning
from src.stages.evaluate import run_evaluate
from src.data.frozen import sha256_file
from src.stages.io import StageError, environment_lock_sha256, read_json, safe_name
from src.stages.prepare import run_prepare
from src.stages.tasks import tuning_tasks
from src.stages.tune import run_tune

SHARD = "ml-quarterly"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    data_dir, manifest_path = synthetic_frozen_copy(tmp_path_factory.mktemp("data"))
    run = tmp_path_factory.mktemp("run")
    bundle = run_prepare(CONFIG, run, data_dir, COMMIT, manifest_path=manifest_path)
    return run, data_dir, manifest_path, bundle


@pytest.fixture
def run(prepared, tmp_path, monkeypatch):
    """A fresh run directory holding the shared prepare bundle."""
    source, data_dir, manifest_path, _ = prepared
    shutil.copytree(source / "prepare", tmp_path / "prepare")
    monkeypatch.setenv("RERUN_CODE_COMMIT", COMMIT)
    return tmp_path, data_dir, manifest_path


def _trials(run_dir, task):
    storage = f"sqlite:///{(run_dir / 'tune' / SHARD / 'studies.sqlite').resolve()}"
    study = optuna.load_study(study_name=tuning.study_name(task.model, task.frequency, task.target), storage=storage)
    return [(t.number, t.params, t.value, t.distributions) for t in study.trials]


def test_worker_processes_give_the_outputs_of_one_process(run, tmp_path):
    """--workers only changes speed: every entry is byte-identical and every archived trial equal."""
    run_dir, data_dir, manifest_path = run
    other = tmp_path / "parallel"
    shutil.copytree(run_dir, other)
    run_tune(run_dir, CONFIG, data_dir, SHARD, manifest_path)
    run_tune(other, CONFIG, data_dir, SHARD, manifest_path, workers=3)
    tasks = [t for t in tuning_tasks() if t.shard == SHARD]
    one, many = (sorted((d / "tune" / SHARD / "tasks").glob("*.json")) for d in (run_dir, other))
    outputs = [p for p in one if not p.name.endswith(".done.json")]
    assert [p.name for p in one] == [p.name for p in many] and len(outputs) == len(tasks)
    for a in outputs:
        assert a.read_bytes() == (other / "tune" / SHARD / "tasks" / a.name).read_bytes(), a.name
    for task in tasks:
        assert _trials(run_dir, task) == _trials(other, task), task.task_id
        record = read_json(other / "tune" / SHARD / "tasks" / f"{safe_name(task.task_id)}.done.json")
        assert record["failed_attempts"] == [] and record["compute_seconds"] > 0
    assert read_json(other / "tune" / SHARD / "heartbeat.json")["done"] == len(tasks)


class FakeStudy:
    """Counts calls per study; ``fail[task_id]`` = number of leading calls that raise."""

    def __init__(self, fail=None):
        self.fail, self.calls = dict(fail or {}), {}

    def __call__(self, model, frequency, target, frame, h, m, settings, seed):
        task_id = f"tune|{model}|{frequency}|{target}"
        self.calls[task_id] = self.calls.get(task_id, 0) + 1
        if self.calls[task_id] <= self.fail.get(task_id, 0):
            raise RuntimeError(f"{task_id} boom")
        study = optuna.create_study(direction="minimize")
        study.add_trial(optuna.trial.create_trial(params={}, distributions={}, value=1.0))
        entry = {"model": model, "frequency": frequency, "target": target, "params": {}, "best_validation_mase": 1.0}
        return entry, tuning.archived_trials(study, task_id)


def test_a_study_is_retried_once_then_listed_and_the_rest_complete(run, monkeypatch):
    run_dir, data_dir, manifest_path = run
    shard_tasks = [t for t in tuning_tasks() if t.shard == SHARD]
    tasks = [t.task_id for t in shard_tasks]
    once, twice = tasks[0], tasks[1]
    fake = FakeStudy({once: 1, twice: 2})
    monkeypatch.setattr(tuning, "run_study", fake)
    with pytest.raises(StageError, match="1 studies failed twice"):
        run_tune(run_dir, CONFIG, data_dir, SHARD, manifest_path)
    root = run_dir / "tune" / SHARD
    assert fake.calls[once] == 2 and fake.calls[twice] == 2
    assert read_json(root / "tasks" / f"{safe_name(once)}.done.json")["failed_attempts"] == [f"RuntimeError: {once} boom"]
    assert not (root / "tasks" / f"{safe_name(twice)}.done.json").exists()
    assert len(read_json(root / "failures" / f"{safe_name(twice)}.json")["attempts"]) == 2
    assert all((root / "tasks" / f"{safe_name(t)}.done.json").exists() for t in tasks if t != twice)

    fake.fail = {}  # a rerun completes the listed study only, and clears its failure
    before = dict(fake.calls)
    run_tune(run_dir, CONFIG, data_dir, SHARD, manifest_path)
    assert {k: v - before.get(k, 0) for k, v in fake.calls.items() if v != before.get(k, 0)} == {twice: 1}
    assert not (root / "failures" / f"{safe_name(twice)}.json").exists()
    t = shard_tasks[1]
    study = f"studies.sqlite:{tuning.study_name(t.model, t.frequency, t.target)}"
    assert read_json(root / "tasks" / f"{safe_name(twice)}.json")["study"] == study
    assert [n for n, *_ in _trials(run_dir, t)] == [0]


@pytest.mark.parametrize("shard", ["neural-quarterly", "transformer-monthly"])
def test_workers_are_refused_on_gpu_shards(run, prepared, shard):
    run_dir, data_dir, manifest_path = run
    provenance = {"code_commit": COMMIT, "environment_lock_sha256": environment_lock_sha256(),
                  "data_manifest_sha256": sha256_file(manifest_path), "bundle_sha256": prepared[3]["bundle_sha256"]}
    tuning.write_frozen_configs(run_dir / "configs_frozen.json", {}, {}, provenance)
    with pytest.raises(StageError, match="--workers is for the ML shards"):
        run_tune(run_dir, CONFIG, data_dir, shard, manifest_path, workers=2)
    with pytest.raises(StageError, match="--workers is for the CPU shards"):
        run_evaluate(run_dir, CONFIG, data_dir, shard, manifest_path, workers=2)
