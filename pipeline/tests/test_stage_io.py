"""Run-directory primitives and the freeze stage (D16, D19, Section 9.4; gates G4; test I4)."""
from __future__ import annotations

import json
import subprocess

import pandas as pd
import pytest

from src.stages import io
from src.stages.io import Heartbeat, StageError, TaskStore
from src.stages.tasks import tuning_tasks
from src.stages.tune import run_freeze

PROV = {"code_commit": "a" * 40, "environment_lock_sha256": "e" * 64, "data_manifest_sha256": "d" * 64, "bundle_sha256": "b" * 64}


def test_code_commit_reads_head_and_refuses_a_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.delenv("RERUN_CODE_COMMIT", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    (repo / "f.txt").write_text("1", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "f.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "c"], check=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert io.code_commit(repo) == head
    (repo / "f.txt").write_text("2", encoding="utf-8")
    with pytest.raises(StageError, match="uncommitted"):
        io.code_commit(repo)
    monkeypatch.setenv("RERUN_CODE_COMMIT", "not-a-hash")
    with pytest.raises(StageError, match="full commit"):
        io.code_commit(repo)


def test_task_store_skips_finished_tasks_and_refuses_mixing(tmp_path):
    store = TaskStore(tmp_path, PROV)
    assert not store.is_done("eval|x")
    sha = io.write_parquet(store.output("eval|x"), pd.DataFrame({"a": [1, 2]}))
    store.complete("eval|x", sha)
    assert store.is_done("eval|x")
    assert not list((tmp_path / "tasks").glob(".*.tmp"))
    with pytest.raises(StageError, match="other provenance"):
        TaskStore(tmp_path, {**PROV, "code_commit": "c" * 40}).is_done("eval|x")
    io.write_parquet(store.output("eval|x"), pd.DataFrame({"a": [9]}))
    with pytest.raises(StageError, match="changed"):
        store.is_done("eval|x")


def test_failures_are_listed_and_cleared_after_a_successful_rerun(tmp_path):
    store = TaskStore(tmp_path, PROV)
    store.record_failure("eval|y", ["RuntimeError: a", "RuntimeError: b"])
    listed = json.loads((tmp_path / "failures" / "eval__y.json").read_text(encoding="utf-8"))
    assert listed["attempts"] == ["RuntimeError: a", "RuntimeError: b"] and listed["provenance"] == PROV
    assert listed["history"] == []
    TaskStore(tmp_path, {**PROV, "code_commit": "b" * 40}).record_failure("eval|y", ["RuntimeError: c"])
    again = store.failure("eval|y")
    assert again["attempts"] == ["RuntimeError: c"] and again["history"][0]["attempts"] == listed["attempts"]
    store.clear_failure("eval|y")
    assert not (tmp_path / "failures" / "eval__y.json").exists()


def test_heartbeat_reports_progress_and_completion(tmp_path):
    path = tmp_path / "heartbeat.json"
    with Heartbeat(path, "evaluate", "ml-monthly", interval=0.05) as beat:
        beat.update(task="eval|z", done=3, total=10)
        state = json.loads(path.read_text(encoding="utf-8"))
        assert (state["task"], state["done"], state["total"], state["shard"]) == ("eval|z", 3, 10, "ml-monthly")
    assert json.loads(path.read_text(encoding="utf-8"))["finished"] is True


def _complete_studies(run, provenance_for=lambda task: PROV, skip=()):
    for task in tuning_tasks():
        if task.task_id in skip:
            continue
        store = TaskStore(run / "tune" / task.shard, provenance_for(task))
        entry = {"model": task.model, "frequency": task.frequency, "target": task.target,
                 "params": {"alpha": 1.0}, "best_validation_mase": 1.0}
        if task.family != "ml":
            entry["trained_steps"] = 500
        io.write_json(store.output(task.task_id, ".json"), entry)
        store.complete(task.task_id, io.sha256_file(store.output(task.task_id, ".json")), suffix=".json")


def test_freeze_collects_all_90_studies_and_never_overwrites(tmp_path):
    _complete_studies(tmp_path)
    digest = run_freeze(tmp_path, {"tuning": {"num_samples": 20}})
    record = json.loads((tmp_path / "configs_frozen.json").read_text(encoding="utf-8"))
    assert record["configs_sha256"] == digest and len(record["entries"]) == 90 and record["provenance"] == PROV
    with pytest.raises(StageError, match="never overwritten"):
        run_freeze(tmp_path, {"tuning": {"num_samples": 20}})


def test_freeze_refuses_missing_studies_or_mixed_provenance(tmp_path):
    _complete_studies(tmp_path / "a", skip={"tune|NHITS|monthly|trend"})
    with pytest.raises(StageError, match="has not completed"):
        run_freeze(tmp_path / "a", {"tuning": {}})
    _complete_studies(tmp_path / "b", provenance_for=lambda t: {**PROV, "code_commit": "f" * 40} if t.family == "ml" else PROV)
    with pytest.raises(StageError, match="different provenance"):
        run_freeze(tmp_path / "b", {"tuning": {}})
