from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd

from src.training import scheduler as sched
from src.training.checkpointing import CheckpointManager
from src.training.parallel_runtime import TaskExecutionResult, TaskWorkerContext, execute_task, initialize_worker
from src.training.train_family import evaluate_task
from src.training.resources import classify_tasks, default_feature_compute_workers, read_cgroup_cpu_quota, safe_task_id, task_output_dir
from src.utils import atomic_write_json, write_dataframe


def _task(task_id: str, family: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "feature_name": "feature_non_normality",
        "frequency": "monthly",
        "decomposition_method": "without_stl",
        "model_family": family,
        "model_name": "AutoETS",
        "finetuning_mode": "no_finetune",
    }


def _context(run_root: Path) -> TaskWorkerContext:
    frame = pd.DataFrame(
        {
            "unique_id": ["s1", "s1", "s1"],
            "ds": pd.date_range("2020-01-01", periods=3, freq="D"),
            "y": [1.0, 2.0, 3.0],
            "source_dataset": ["M3_Monthly"] * 3,
        }
    )
    return TaskWorkerContext(
        run_root=str(run_root),
        season_length=12,
        horizon=1,
        optuna_num_samples=1,
        threads_per_cpu_task=1,
        threads_per_gpu_task=2,
        data_by_frequency={"monthly": frame},
        cutoffs_by_frequency={"monthly": pd.DataFrame({"unique_id": ["s1"], "train_end_idx": [2], "window": [0], "cutoff": [1]})},
        bucket_manifest=pd.DataFrame({"feature_name": ["feature_non_normality"], "unique_id": ["s1"], "feature_tercile_bucket": ["Low"]}),
    )


def _evaluation_inputs(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = _context(run_root)
    return context.data_by_frequency["monthly"], context.cutoffs_by_frequency["monthly"], context.bucket_manifest


def _family_task(family: str, model_name: str) -> dict[str, object]:
    task = _task(family[:3] + "|task", family)
    task["model_name"] = model_name
    return task


def test_read_cgroup_cpu_quota_parses_common_forms(tmp_path: Path):
    quota = tmp_path / "cpu.max"
    quota.write_text("max 100000", encoding="utf-8")
    parsed = read_cgroup_cpu_quota(quota)
    assert parsed.quota_raw == "max 100000"
    assert parsed.effective_cores == float(__import__("os").cpu_count() or 1)

    quota.write_text("960000 100000", encoding="utf-8")
    parsed = read_cgroup_cpu_quota(quota)
    assert parsed.period == 100000
    assert np.isclose(parsed.effective_cores, 9.6)

    parsed = read_cgroup_cpu_quota(tmp_path / "missing.cpu.max")
    assert parsed.effective_cores == float(__import__("os").cpu_count() or 1)


def test_default_feature_compute_workers_uses_effective_cores(monkeypatch):
    monkeypatch.setattr("src.training.resources.detect_effective_cpu_cores", lambda: 9.6)
    assert default_feature_compute_workers() == 8


def test_task_classification_and_safe_paths(tmp_path: Path):
    tasks = classify_tasks([
        _task("stat|1", "statistical"),
        _task("ml|1", "mlforecast"),
        _task("neural|1", "neuralforecast"),
    ])
    assert [t["resource_class"] for t in tasks] == ["cpu", "cpu", "gpu"]
    assert safe_task_id("a|b/c") == "a_b_c"
    assert task_output_dir(tmp_path, "a|b/c").parent == tmp_path / "task_outputs"


class ImmediateExecutor:
    def __init__(self, max_workers: int = 1, initializer=None, initargs=(), **kwargs):
        self.max_workers = max_workers
        if initializer is not None:
            initializer(*initargs)

    def submit(self, fn, *args, **kwargs):
        fut = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - defensive
            fut.set_exception(exc)
        return fut

    def shutdown(self, wait: bool = True, cancel_futures: bool = False):
        return None


def _fake_execute_task_factory(run_root: Path):
    def _fake(task: dict[str, object]) -> TaskExecutionResult:
        tdir = task_output_dir(run_root, str(task["task_id"]))
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "optuna").mkdir(parents=True, exist_ok=True)
        metrics = pd.DataFrame([{ "task_id": task["task_id"], "unique_id": "s1", "window": 0, "rel_naive_unclipped": 0.5, "rel_naive_clipped": 0.5, "mae": 1.0, "smape": 1.0, "mase": 1.0, "wape": 1.0, "model_output_source": "placeholder", "fallback_reason": "test", "training_time_seconds": 1.0, "inference_time_seconds": 0.1 }])
        forecasts = pd.DataFrame([{ "task_id": task["task_id"], "unique_id": "s1", "ds": pd.Timestamp("2020-01-01"), "y": 1.0, "yhat": 1.1, "model_output_source": "placeholder", "fallback_reason": "test" }])
        naive = pd.DataFrame([{ "task_id": task["task_id"], "unique_id": "s1", "ds": pd.Timestamp("2020-01-01"), "y": 1.0, "yhat_naive": 1.0, "model_output_source": "placeholder", "fallback_reason": "test" }])
        write_dataframe(tdir / "metrics.parquet", metrics)
        write_dataframe(tdir / "forecasts.parquet", forecasts)
        write_dataframe(tdir / "naive_forecasts.parquet", naive)
        atomic_write_json(tdir / "DONE.json", {"task_id": task["task_id"], "training_time_seconds": 1.0, "inference_time_seconds": 0.1, "training_seconds": 1.0, "inference_seconds": 0.1, "status": "done", "model_output_source": "placeholder", "fallback_reason": "test", "started_at_utc": "2026-05-21T00:00:00Z", "finished_at_utc": "2026-05-21T00:00:01Z", "output_dir": str(tdir)})
        return TaskExecutionResult(
            task_id=str(task["task_id"]),
            safe_task_id=safe_task_id(str(task["task_id"])),
            task_dir=str(tdir),
            resource_class=str(task.get("resource_class", "cpu")),
            status="done",
            elapsed_seconds=1.0,
            training_time_seconds=1.0,
            inference_time_seconds=0.1,
            started_at_utc="2026-05-21T00:00:00Z",
            finished_at_utc="2026-05-21T00:00:01Z",
            output_dir=str(tdir),
            model_output_source="placeholder",
            fallback_reason="test",
            metrics_path=str(tdir / "metrics.parquet"),
            forecasts_path=str(tdir / "forecasts.parquet"),
            naive_path=str(tdir / "naive_forecasts.parquet"),
        )

    return _fake


def test_parallel_schedule_with_dummy_executor_skips_completed_and_writes_checkpoints(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "run"
    run_root.mkdir()
    cp = CheckpointManager(run_root, "run_1")
    tasks = [
        _task("done_task", "statistical"),
        _task("new_task", "neuralforecast"),
    ]
    done_dir = task_output_dir(run_root, "done_task")
    done_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(done_dir / "DONE.json", {"task_id": "done_task", "training_seconds": 2.0, "inference_seconds": 0.2, "status": "done"})
    write_dataframe(done_dir / "metrics.parquet", pd.DataFrame([{"task_id": "done_task", "unique_id": "s1", "window": 0, "rel_naive_unclipped": 0.5, "rel_naive_clipped": 0.5, "mae": 1.0, "smape": 1.0, "mase": 1.0, "wape": 1.0}]))
    write_dataframe(done_dir / "forecasts.parquet", pd.DataFrame([{"task_id": "done_task", "unique_id": "s1", "ds": pd.Timestamp("2020-01-01"), "y": 1.0, "yhat": 1.0}]))
    write_dataframe(done_dir / "naive_forecasts.parquet", pd.DataFrame([{"task_id": "done_task", "unique_id": "s1", "ds": pd.Timestamp("2020-01-01"), "y": 1.0, "yhat_naive": 1.0}]))

    monkeypatch.setattr(sched, "execute_task", _fake_execute_task_factory(run_root))
    monkeypatch.setattr(sched.futures, "ProcessPoolExecutor", ImmediateExecutor)

    result = sched.run_parallel_schedule(
        run_root=run_root,
        tasks=tasks,
        data_by_frequency={"monthly": _context(run_root).data_by_frequency["monthly"]},
        cutoffs_by_frequency={"monthly": _context(run_root).cutoffs_by_frequency["monthly"]},
        bucket_manifest=_context(run_root).bucket_manifest,
        season_length_by_frequency={"monthly": 12},
        horizon_by_frequency={"monthly": 1},
        parallel_config={
            "forecast_parallel_enabled": True,
            "forecast_cpu_task_workers": 2,
            "forecast_gpu_task_workers": 1,
            "forecast_statistical_workers": 1,
            "forecast_mlforecast_workers": 1,
            "forecast_neural_workers": 1,
            "forecast_transformer_workers": 1,
            "forecast_threads_per_cpu_task": 1,
            "forecast_threads_per_gpu_task": 2,
            "effective_cpu_cores": 9.6,
            "cpu_max": "960000 100000",
            "cpu_period": 100000,
        },
        optuna_num_samples=1,
        cp=cp,
        logger=type("Logger", (), {"error": lambda *args, **kwargs: None})(),
        dry_run=False,
    )

    assert cp.is_done("new_task")
    assert (run_root / "checkpoints" / "completed_tasks.parquet").exists()
    completed = pd.read_parquet(run_root / "checkpoints" / "completed_tasks.parquet")
    required = {
        "task_id",
        "status",
        "model_family",
        "model_name",
        "decomposition_method",
        "finetuning_mode",
        "feature_name",
        "frequency",
        "started_at_utc",
        "finished_at_utc",
        "training_time_seconds",
        "inference_time_seconds",
        "output_dir",
        "fallback_reason",
    }
    assert required.issubset(set(completed.columns))
    assert completed["training_time_seconds"].fillna(0).gt(0).any()
    assert completed["inference_time_seconds"].fillna(0).gt(0).any()
    assert set(result["completed_ids"]) >= {"done_task", "new_task"}
    assert not result["failed"]
    assert result["metrics"].shape[0] == 2


def test_sequential_schedule_still_works(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "run_seq"
    run_root.mkdir()
    cp = CheckpointManager(run_root, "run_2")
    tasks = [_task("seq_task", "statistical")]
    monkeypatch.setattr(sched, "execute_task", _fake_execute_task_factory(run_root))

    result = sched.run_parallel_schedule(
        run_root=run_root,
        tasks=tasks,
        data_by_frequency={"monthly": _context(run_root).data_by_frequency["monthly"]},
        cutoffs_by_frequency={"monthly": _context(run_root).cutoffs_by_frequency["monthly"]},
        bucket_manifest=_context(run_root).bucket_manifest,
        season_length_by_frequency={"monthly": 12},
        horizon_by_frequency={"monthly": 1},
        parallel_config={"forecast_parallel_enabled": False, "effective_cpu_cores": 9.6},
        optuna_num_samples=1,
        cp=cp,
        logger=type("Logger", (), {"error": lambda *args, **kwargs: None})(),
        dry_run=False,
    )

    assert cp.is_done("seq_task")
    assert result["metrics"].shape[0] == 1


def test_worker_failure_creates_failed_json(tmp_path: Path, monkeypatch):
    run_root = tmp_path / "worker_fail"
    run_root.mkdir()
    initialize_worker(_context(run_root))

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.training.parallel_runtime.evaluate_task", boom)
    result = execute_task(_task("bad_task", "statistical"))

    assert result.status == "failed"
    assert (task_output_dir(run_root, "bad_task") / "FAILED.json").exists()


def test_evaluate_task_emits_timings_and_sources(tmp_path: Path):
    df, cutoffs, bucket_manifest = _evaluation_inputs(tmp_path)
    task = _task("stat|audit", "statistical")
    task["model_name"] = "AutoARIMA"
    evaluation = evaluate_task(task, df, cutoffs, season_length=12, horizon=1, bucket_manifest=bucket_manifest)

    assert evaluation.training_time_seconds > 0
    assert evaluation.inference_time_seconds > 0
    assert not evaluation.metrics.empty
    assert {"model_output_source", "fallback_reason", "training_time_seconds", "inference_time_seconds"}.issubset(evaluation.metrics.columns)
    assert evaluation.forecasts["yhat"].notna().all()


def test_neural_and_transformer_tasks_declare_fallback_or_trained_model(tmp_path: Path):
    from src.models.forecast_result import ALLOWED_OUTPUT_SOURCES
    df, cutoffs, bucket_manifest = _evaluation_inputs(tmp_path)
    for family, model_name in [("neuralforecast", "AutoNLinear"), ("transformers", "AutoTFT")]:
        task = _task(f"{family[:3]}|audit", family)
        task["model_name"] = model_name
        evaluation = evaluate_task(task, df, cutoffs, season_length=12, horizon=1, bucket_manifest=bucket_manifest)
        assert evaluation.model_output_source in ALLOWED_OUTPUT_SOURCES, (
            f"Unexpected model_output_source: {evaluation.model_output_source!r}"
        )
        # Non-trained results must carry a fallback reason
        if not evaluation.model_output_source.startswith("trained_"):
            assert evaluation.fallback_reason


def test_model_families_do_not_all_collapse_to_identical_forecasts(tmp_path: Path):
    df, cutoffs, bucket_manifest = _evaluation_inputs(tmp_path)
    fixtures = [
        ("statistical", "AutoARIMA"),
        ("mlforecast", "AutoRidge"),
        ("neuralforecast", "AutoNLinear"),
        ("transformers", "AutoTFT"),
    ]
    hashes: dict[str, tuple[float, ...]] = {}
    for family, model_name in fixtures:
        task = _task(f"{family[:3]}|hash", family)
        task["model_name"] = model_name
        evaluation = evaluate_task(task, df, cutoffs, season_length=12, horizon=1, bucket_manifest=bucket_manifest)
        hashes[family] = tuple(np.round(evaluation.forecasts["yhat"].head(8).to_numpy(dtype=float), 6))

    assert len(set(hashes.values())) > 1
