"""Stage R3 evaluate: order, resume, retry-once and failure listing (D16, D19, Section 9.4; test I4).

The engine is replaced by stand-ins that count calls, so the test checks the stage's
bookkeeping on a real prepare bundle; the engine itself is covered by test_engine.
"""
from __future__ import annotations

import copy
import json
import shutil

import pandas as pd
import pytest

from helpers_run import COMMIT, CONFIG, synthetic_frozen_copy
from src.data.frozen import sha256_file
from src.forecast import engine, tuning
from src.stages import evaluate
from src.stages.evaluate import run_evaluate
from src.stages.io import StageError, environment_lock_sha256, read_json, safe_name
from src.stages.prepare import run_prepare
from src.stages.tasks import evaluation_tasks, statistical_chunk_members


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    data_dir, manifest_path = synthetic_frozen_copy(tmp_path_factory.mktemp("data"))
    run = tmp_path_factory.mktemp("run")
    bundle = run_prepare(CONFIG, run, data_dir, COMMIT, manifest_path=manifest_path)
    return run, data_dir, manifest_path, bundle


@pytest.fixture
def run(prepared, tmp_path, monkeypatch):
    """A fresh run directory: the shared prepare bundle plus frozen configurations."""
    source, data_dir, manifest_path, bundle = prepared
    shutil.copytree(source / "prepare", tmp_path / "prepare")
    provenance = {"code_commit": COMMIT, "environment_lock_sha256": environment_lock_sha256(),
                  "data_manifest_sha256": sha256_file(manifest_path), "bundle_sha256": bundle["bundle_sha256"]}
    tuning.write_frozen_configs(tmp_path / "configs_frozen.json", {}, {}, provenance)
    monkeypatch.setenv("RERUN_CODE_COMMIT", COMMIT)
    return tmp_path, data_dir, manifest_path


def _shard_tasks(run_dir, shard):
    prepare = run_dir / "prepare"
    tasks = evaluation_tasks(pd.read_parquet(prepare / "bucket_summary.parquet"), pd.read_parquet(prepare / "samples.parquet"),
                             CONFIG["random_seed"], CONFIG["seed_check_seeds"])
    return [t for t in tasks if t.shard == shard]


def _spec_id(spec):
    return f"eval|{spec['feature_name']}|{spec['frequency']}|{spec['strategy']}|{spec['model']}|{spec['scope']}|seed{spec['seed']}"


class FakeGlobal:
    """Counts calls per task; ``fail[task_id]`` = number of leading calls that raise."""

    def __init__(self, fail=None):
        self.fail, self.calls = dict(fail or {}), {}

    def __call__(self, spec, pool, cutoffs, entries, bucket_of, h, m, prov):
        task_id = _spec_id(spec)
        self.calls[task_id] = self.calls.get(task_id, 0) + 1
        if self.calls[task_id] <= self.fail.get(task_id, 0):
            raise RuntimeError(f"induced failure {self.calls[task_id]}")
        assert set(pool["unique_id"]) == set(bucket_of)
        return pd.DataFrame({"unique_id": sorted(bucket_of), "status": "trained"})


def test_global_shard_retries_once_lists_second_failures_and_resumes(run, monkeypatch):
    run_dir, data_dir, manifest_path = run
    tasks = _shard_tasks(run_dir, "ml-quarterly")
    assert [t.priority for t in tasks] == sorted(t.priority for t in tasks)
    flaky = tasks[0].task_id
    broken = next(t.task_id for t in tasks if t.priority == 1)
    fake = FakeGlobal({flaky: 1, broken: 2})
    monkeypatch.setattr(engine, "evaluate_global_task", fake)

    with pytest.raises(StageError, match="1 global tasks failed twice"):
        run_evaluate(run_dir, CONFIG, data_dir, "ml-quarterly", manifest_path)
    root = run_dir / "evaluate" / "ml-quarterly"
    assert fake.calls[flaky] == 2 and fake.calls[broken] == 2
    assert all(n == 1 for task_id, n in fake.calls.items() if task_id not in (flaky, broken))
    assert set(fake.calls) == {t.task_id for t in tasks}  # the stage ran every task after the failure
    failure = read_json(root / "failures" / f"{safe_name(broken)}.json")
    assert failure["attempts"] == ["RuntimeError: induced failure 1", "RuntimeError: induced failure 2"]
    assert failure["provenance"]["code_commit"] == COMMIT
    assert not (root / "failures" / f"{safe_name(flaky)}.json").exists()
    assert not (root / "tasks" / f"{safe_name(broken)}.done.json").exists()
    done = read_json(root / "tasks" / f"{safe_name(flaky)}.done.json")
    out = pd.read_parquet(root / "tasks" / done["file"])
    assert (out["task_id"] == flaky).all() and done["rows"] == len(out) and done["failed_rows"] == 0
    beat = read_json(root / "heartbeat.json")
    assert (beat["finished"], beat["done"], beat["total"], beat["failed"]) == (True, len(tasks), len(tasks), 1)

    fake.fail.clear()  # the cause is fixed; rerun only the listed task (D16)
    fake.calls.clear()
    assert run_evaluate(run_dir, CONFIG, data_dir, "ml-quarterly", manifest_path, only={broken}) == {"shard": "ml-quarterly", "tasks": 1}
    assert fake.calls == {broken: 1}
    assert not (root / "failures" / f"{safe_name(broken)}.json").exists()
    resolved = read_json(root / "failures" / "resolved" / f"{safe_name(broken)}.json")
    assert resolved["resolved_by_commit"] == COMMIT and len(resolved["attempts"]) == 2
    assert not (root / "d16_reruns.json").exists()  # same commit: a plain rerun, not a fix
    assert read_json(root / "tasks" / f"{safe_name(flaky)}.done.json")["failed_attempts"] == ["RuntimeError: induced failure 1"]

    fake.calls.clear()  # a complete shard resumes to nothing
    run_evaluate(run_dir, CONFIG, data_dir, "ml-quarterly", manifest_path)
    assert fake.calls == {}


def test_statistical_chunks_record_per_series_failures_as_rows(run, monkeypatch):
    run_dir, data_dir, manifest_path = run
    samples = pd.read_parquet(run_dir / "prepare" / "samples.parquet")
    members = statistical_chunk_members(samples, "quarterly", 0)
    calls = []

    def fake(model, frequency, strategy, series, cutoffs, h, m, prov):
        uid = series["unique_id"].iloc[0]
        calls.append((model, strategy, uid))
        assert series["unique_id"].nunique() == 1 and (cutoffs["frequency"] == "quarterly").all() and (h, m) == (8, 4)
        return pd.DataFrame({"unique_id": [uid] * 3, "window": [0, 1, 2], "status": ["failed" if uid == members[0] else "trained"] * 3})

    monkeypatch.setattr(engine, "evaluate_statistical_series", fake)
    run_evaluate(run_dir, CONFIG, data_dir, "statistical-quarterly", manifest_path)
    tasks = _shard_tasks(run_dir, "statistical-quarterly")
    assert len(tasks) == 9 and len(calls) == 9 * len(members)
    for task in tasks:
        done = read_json(run_dir / "evaluate" / "statistical-quarterly" / "tasks" / f"{safe_name(task.task_id)}.done.json")
        assert (done["rows"], done["failed_rows"]) == (3 * len(members), 3)


def test_evaluate_refuses_another_config_bundle_or_changed_prepare_files(run):
    run_dir, data_dir, manifest_path = run
    changed = copy.deepcopy(CONFIG)
    changed["tuning_set"]["n_per_source"] = 7
    with pytest.raises(StageError, match="configuration differs"):
        run_evaluate(run_dir, changed, data_dir, "ml-quarterly", manifest_path)

    record = json.loads((run_dir / "configs_frozen.json").read_text(encoding="utf-8"))
    (run_dir / "configs_frozen.json").unlink()
    tuning.write_frozen_configs(run_dir / "configs_frozen.json", {}, {}, {**record["provenance"], "bundle_sha256": "0" * 64})
    with pytest.raises(StageError, match="another prepare bundle"):
        run_evaluate(run_dir, CONFIG, data_dir, "ml-quarterly", manifest_path)

    cutoffs = pd.read_parquet(run_dir / "prepare" / "cutoffs.parquet")
    cutoffs.iloc[:-1].to_parquet(run_dir / "prepare" / "cutoffs.parquet", index=False)
    with pytest.raises(StageError, match="cutoffs.parquet changed"):
        run_evaluate(run_dir, CONFIG, data_dir, "ml-quarterly", manifest_path)


def test_another_commit_only_reruns_listed_failures_after_a_descendant_fix(run, monkeypatch):
    """D16: the fix commit must descend from the frozen one; the rerun is registered for G4."""
    run_dir, data_dir, manifest_path = run
    tasks = _shard_tasks(run_dir, "ml-quarterly")
    broken, other = tasks[0].task_id, tasks[1].task_id
    fake = FakeGlobal({broken: 2})
    monkeypatch.setattr(engine, "evaluate_global_task", fake)
    with pytest.raises(StageError, match="failed twice"):
        run_evaluate(run_dir, CONFIG, data_dir, "ml-quarterly", manifest_path)

    fix = "b" * 40
    monkeypatch.setenv("RERUN_CODE_COMMIT", fix)
    fake.fail.clear()
    with pytest.raises(StageError, match="another commit only reruns listed failures"):
        run_evaluate(run_dir, CONFIG, data_dir, "ml-quarterly", manifest_path)
    monkeypatch.setattr(evaluate, "is_ancestor", lambda old, new: False)
    with pytest.raises(StageError, match="does not descend"):
        run_evaluate(run_dir, CONFIG, data_dir, "ml-quarterly", manifest_path, only={broken})
    monkeypatch.setattr(evaluate, "is_ancestor", lambda old, new: (old, new) == (COMMIT, fix))
    with pytest.raises(StageError, match="not a listed failure"):
        run_evaluate(run_dir, CONFIG, data_dir, "ml-quarterly", manifest_path, only={broken, other})
    with pytest.raises(StageError, match="not tasks of shard"):
        run_evaluate(run_dir, CONFIG, data_dir, "ml-quarterly", manifest_path, only={broken, "eval|nonexistent"})

    fake.calls.clear()
    run_evaluate(run_dir, CONFIG, data_dir, "ml-quarterly", manifest_path, only={broken})
    root = run_dir / "evaluate" / "ml-quarterly"
    assert fake.calls == {broken: 1}
    register = read_json(root / "d16_reruns.json")
    assert set(register) == {broken}
    assert (register[broken]["main_commit"], register[broken]["fix_commit"]) == (COMMIT, fix)
    assert register[broken]["failure"]["provenance"]["code_commit"] == COMMIT
    assert read_json(root / "tasks" / f"{safe_name(broken)}.done.json")["provenance"]["code_commit"] == fix
    assert read_json(root / "tasks" / f"{safe_name(other)}.done.json")["provenance"]["code_commit"] == COMMIT


def test_evaluate_refuses_another_environment_than_tuning(run, monkeypatch):
    run_dir, data_dir, manifest_path = run
    monkeypatch.setattr(evaluate, "evaluation_provenance",
                        lambda *a: engine.Provenance(COMMIT, "f" * 64, sha256_file(manifest_path), "x", "y"))
    with pytest.raises(StageError, match="environment_lock_sha256 differs"):
        run_evaluate(run_dir, CONFIG, data_dir, "ml-quarterly", manifest_path)


@pytest.mark.parametrize("shard", ["statistical-quarterly", "ml-quarterly"])
def test_worker_processes_give_the_outputs_of_one_process(run, tmp_path, monkeypatch, shard):
    """--workers only changes speed: every task output is byte-identical to a one-process run."""
    from helpers_run import fake_fits

    from src.forecast.registry import config_key
    from src.stages.tasks import tuning_tasks

    run_dir, data_dir, manifest_path = run
    fake_fits(monkeypatch)  # inherited by the forked workers
    record = read_json(run_dir / "configs_frozen.json")
    entries = {config_key(t.model, t.frequency, t.target): {"model": t.model, "frequency": t.frequency, "target": t.target,
                                                             "params": {}, **({} if t.family == "ml" else {"trained_steps": 1})}
               for t in tuning_tasks()}
    (run_dir / "configs_frozen.json").unlink()
    tuning.write_frozen_configs(run_dir / "configs_frozen.json", entries, {}, record["provenance"])
    only = {t.task_id for t in _shard_tasks(run_dir, shard)[:20]}
    other = tmp_path / "parallel"
    shutil.copytree(run_dir, other)
    run_evaluate(run_dir, CONFIG, data_dir, shard, manifest_path, only=only)
    run_evaluate(other, CONFIG, data_dir, shard, manifest_path, only=only, workers=3)
    one, many = (sorted((d / "evaluate" / shard / "tasks").glob("*.parquet")) for d in (run_dir, other))
    assert [p.name for p in one] == [p.name for p in many] and len(one) == len(only)
    for a, b in zip(one, many):
        assert a.read_bytes() == b.read_bytes(), a.name
    assert read_json(other / "evaluate" / shard / "heartbeat.json")["done"] == len(one)
