from __future__ import annotations

import os
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.features.scaling import fit_transform_features  # noqa: F401  # imported for worker-side module availability
from src.training.task_graph import BUDGETS  # noqa: F401
from src.training.train_family import evaluate_task
from src.training.strict_mode import StrictModeViolation
from src.training.tune import write_optuna_artifacts
from src.utils import atomic_write_json, utc_now_iso, write_dataframe

from .resources import safe_task_id, task_output_dir, task_resource_class


@dataclass(frozen=True)
class TaskWorkerContext:
    run_root: str
    season_length: int
    horizon: int
    optuna_num_samples: int
    threads_per_cpu_task: int
    threads_per_gpu_task: int
    data_by_frequency: dict[str, pd.DataFrame]
    cutoffs_by_frequency: dict[str, pd.DataFrame]
    bucket_manifest: pd.DataFrame


@dataclass(frozen=True)
class TaskExecutionResult:
    task_id: str
    safe_task_id: str
    task_dir: str
    resource_class: str
    status: str
    elapsed_seconds: float
    training_time_seconds: float | None = None
    inference_time_seconds: float | None = None
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    output_dir: str | None = None
    model_output_source: str | None = None
    fallback_reason: str | None = None
    metrics_path: str | None = None
    forecasts_path: str | None = None
    naive_path: str | None = None
    error: str | None = None
    traceback_text: str | None = None


_WORKER_CONTEXT: TaskWorkerContext | None = None


def initialize_worker(context: TaskWorkerContext) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _set_thread_limits(task: dict[str, object], context: TaskWorkerContext) -> None:
    resource_class = task_resource_class(task)
    threads = context.threads_per_cpu_task if resource_class == "cpu" else context.threads_per_gpu_task
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = str(max(1, int(threads)))


def _task_dir(context: TaskWorkerContext, task_id: str) -> Path:
    return task_output_dir(Path(context.run_root), task_id)


def execute_task(task: dict[str, object]) -> TaskExecutionResult:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context was not initialized")
    context = _WORKER_CONTEXT
    task_id = str(task["task_id"])
    safe_id = safe_task_id(task_id)
    tdir = _task_dir(context, task_id)
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "optuna").mkdir(parents=True, exist_ok=True)
    _set_thread_limits(task, context)
    started_at_utc = utc_now_iso()

    if (tdir / "DONE.json").exists():
        payload = {}
        try:
            import json

            payload = json.loads((tdir / "DONE.json").read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        training_time_seconds = float(payload.get("training_time_seconds", payload.get("training_seconds", 0.0)) or 0.0)
        inference_time_seconds = float(payload.get("inference_time_seconds", payload.get("inference_seconds", 0.0)) or 0.0)
        started_at = payload.get("started_at_utc") or started_at_utc
        finished_at = payload.get("finished_at_utc") or utc_now_iso()
        return TaskExecutionResult(
            task_id=task_id,
            safe_task_id=safe_id,
            task_dir=str(tdir),
            resource_class=task_resource_class(task),
            status="already_done",
            elapsed_seconds=float(training_time_seconds + inference_time_seconds),
            training_time_seconds=training_time_seconds,
            inference_time_seconds=inference_time_seconds,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            output_dir=str(tdir),
            model_output_source=str(payload.get("model_output_source", "placeholder")),
            fallback_reason=payload.get("fallback_reason"),
            metrics_path=str(tdir / "metrics.parquet"),
            forecasts_path=str(tdir / "forecasts.parquet"),
            naive_path=str(tdir / "naive_forecasts.parquet"),
        )

    try:
        freq = str(task["frequency"])
        evaluation = evaluate_task(
            task,
            context.data_by_frequency[freq],
            context.cutoffs_by_frequency[freq],
            int(context.season_length),
            int(context.horizon),
            context.bucket_manifest,
            num_optuna_samples=int(context.optuna_num_samples),
        )
        finished_at_utc = evaluation.finished_at_utc or utc_now_iso()
        write_dataframe(tdir / "metrics.parquet", evaluation.metrics)
        write_dataframe(tdir / "forecasts.parquet", evaluation.forecasts)
        write_dataframe(tdir / "naive_forecasts.parquet", evaluation.naive)
        write_optuna_artifacts(tdir, str(task.get("model_name", "model")), int(context.optuna_num_samples), tuning_status="disabled_by_budget")
        done_payload = {
            **task,
            "task_id": task_id,
            "safe_task_id": safe_id,
            "resource_class": task_resource_class(task),
            "status": "done",
            "started_at_utc": evaluation.started_at_utc or started_at_utc,
            "finished_at_utc": finished_at_utc,
            "output_dir": str(tdir),
            "model_output_source": evaluation.model_output_source,
            "fallback_reason": evaluation.fallback_reason,
            "model_backend": evaluation.model_backend,
            "backend_library_version": evaluation.backend_library_version,
            "forecast_hash": evaluation.forecast_hash,
            "training_time_seconds": evaluation.training_time_seconds,
            "inference_time_seconds": evaluation.inference_time_seconds,
            "training_seconds": evaluation.training_time_seconds,
            "inference_seconds": evaluation.inference_time_seconds,
            "metrics_rows": int(len(evaluation.metrics)),
            "forecast_rows": int(len(evaluation.forecasts)),
        }
        atomic_write_json(tdir / "DONE.json", done_payload)
        return TaskExecutionResult(
            task_id=task_id,
            safe_task_id=safe_id,
            task_dir=str(tdir),
            resource_class=task_resource_class(task),
            status="done",
            elapsed_seconds=float((evaluation.training_time_seconds or 0.0) + (evaluation.inference_time_seconds or 0.0)),
            training_time_seconds=float(evaluation.training_time_seconds),
            inference_time_seconds=float(evaluation.inference_time_seconds),
            started_at_utc=evaluation.started_at_utc,
            finished_at_utc=finished_at_utc,
            output_dir=str(tdir),
            model_output_source=evaluation.model_output_source,
            fallback_reason=evaluation.fallback_reason,
            metrics_path=str(tdir / "metrics.parquet"),
            forecasts_path=str(tdir / "forecasts.parquet"),
            naive_path=str(tdir / "naive_forecasts.parquet"),
        )
    except Exception as exc:
        tb = traceback.format_exc()
        output_source = "strict_mode_failure" if isinstance(exc, StrictModeViolation) else "failed_fallback"
        failed_payload = {
            **task,
            "task_id": task_id,
            "safe_task_id": safe_id,
            "resource_class": task_resource_class(task),
            "status": "failed",
            "error": repr(exc),
            "traceback": tb,
            "started_at_utc": started_at_utc,
            "finished_at_utc": utc_now_iso(),
            "output_dir": str(tdir),
            "model_output_source": output_source,
            "fallback_reason": f"worker_exception:{exc.__class__.__name__}",
        }
        atomic_write_json(tdir / "FAILED.json", failed_payload)
        return TaskExecutionResult(
            task_id=task_id,
            safe_task_id=safe_id,
            task_dir=str(tdir),
            resource_class=task_resource_class(task),
            status="failed",
            elapsed_seconds=0.0,
            started_at_utc=started_at_utc,
            finished_at_utc=utc_now_iso(),
            output_dir=str(tdir),
            model_output_source=output_source,
            fallback_reason=f"worker_exception:{exc.__class__.__name__}",
            error=repr(exc),
            traceback_text=tb,
        )
