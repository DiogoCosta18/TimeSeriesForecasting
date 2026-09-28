from __future__ import annotations

import concurrent.futures as futures
import multiprocessing
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils import atomic_write_json, write_dataframe

from .checkpointing import CheckpointManager
from .parallel_runtime import TaskExecutionResult, TaskWorkerContext, execute_task, initialize_worker
from .resources import classify_tasks, safe_task_id, summarize_task_plan, task_output_dir, task_resource_class


def _read_task_frame(task_dir: Path, filename: str) -> pd.DataFrame:
    path = task_dir / filename
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            csv = path.with_suffix(".csv")
            if csv.exists():
                return pd.read_csv(csv)
    return pd.DataFrame()


def _read_done_payload(task_dir: Path) -> dict[str, object]:
    path = task_dir / "DONE.json"
    if not path.exists():
        return {}
    try:
        return pd.read_json(path, typ="series").to_dict()
    except Exception:
        try:
            import json

            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


def _load_completed_task_outputs(run_root: Path, tasks: list[dict[str, object]]) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], set[str]]:
    metrics_parts: list[pd.DataFrame] = []
    forecast_parts: list[pd.DataFrame] = []
    naive_parts: list[pd.DataFrame] = []
    completed: set[str] = set()
    for task in tasks:
        task_id = str(task["task_id"])
        tdir = task_output_dir(run_root, task_id)
        if (tdir / "DONE.json").exists():
            completed.add(task_id)
            metrics_parts.append(_read_task_frame(tdir, "metrics.parquet"))
            forecast_parts.append(_read_task_frame(tdir, "forecasts.parquet"))
            naive_parts.append(_read_task_frame(tdir, "naive_forecasts.parquet"))
    return metrics_parts, forecast_parts, naive_parts, completed


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _write_merged_outputs(run_root: Path, metrics_parts: list[pd.DataFrame], forecast_parts: list[pd.DataFrame], naive_parts: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = _concat(metrics_parts)
    forecasts = _concat(forecast_parts)
    naive = _concat(naive_parts)
    write_dataframe(run_root / "cv" / "fold_metrics.parquet", metrics)
    write_dataframe(run_root / "cv" / "fold_forecasts.parquet", forecasts)
    write_dataframe(run_root / "cv" / "fold_naive_forecasts.parquet", naive)
    return metrics, forecasts, naive


def _task_record(task: dict[str, object], result: TaskExecutionResult | dict[str, object]) -> dict[str, object]:
    base = dict(task)
    if isinstance(result, TaskExecutionResult):
        base.update(
            {
                "status": result.status,
                "started_at_utc": result.started_at_utc,
                "finished_at_utc": result.finished_at_utc,
                "training_time_seconds": result.training_time_seconds,
                "inference_time_seconds": result.inference_time_seconds,
                "output_dir": result.output_dir or result.task_dir,
                "fallback_reason": result.fallback_reason,
                "model_output_source": result.model_output_source,
            }
        )
    else:
        base.update(result)
    base.setdefault("status", "done")
    base.setdefault("training_time_seconds", 0.0)
    base.setdefault("inference_time_seconds", 0.0)
    if not base.get("output_dir"):
        base["output_dir"] = base.get("task_dir") or base.get("output_dir")
    return base


def _submit_task(executor: futures.Executor, task: dict[str, object]) -> futures.Future:
    return executor.submit(execute_task, task)


def _schedule_queue(
    executor: futures.Executor | None,
    pending: dict[str, deque[dict[str, object]]],
    active: dict[futures.Future, dict[str, object]],
    family_limits: dict[str, int],
    family_active: Counter,
    total_limit: int,
) -> None:
    if executor is None:
        return
    while len(active) < total_limit:
        submitted = False
        for family, queue in pending.items():
            if not queue:
                continue
            if family_active[family] >= family_limits.get(family, total_limit):
                continue
            task = queue.popleft()
            fut = _submit_task(executor, task)
            active[fut] = task
            family_active[family] += 1
            submitted = True
            if len(active) >= total_limit:
                break
        if not submitted:
            break


def build_parallel_plan(tasks: list[dict[str, object]], completed_ids: set[str]) -> dict[str, object]:
    classified = classify_tasks(tasks)
    plan = summarize_task_plan(classified, completed_ids)
    plan["tasks"] = classified
    plan["pending_tasks"] = [task for task in classified if str(task["task_id"]) not in completed_ids]
    plan["cpu_tasks"] = [task for task in plan["pending_tasks"] if task_resource_class(task) == "cpu"]
    plan["gpu_tasks"] = [task for task in plan["pending_tasks"] if task_resource_class(task) == "gpu"]
    return plan


def run_parallel_schedule(
    run_root: Path,
    tasks: list[dict[str, object]],
    data_by_frequency: dict[str, pd.DataFrame],
    cutoffs_by_frequency: dict[str, pd.DataFrame],
    bucket_manifest: pd.DataFrame,
    season_length_by_frequency: dict[str, int],
    horizon_by_frequency: dict[str, int],
    parallel_config: dict[str, object],
    optuna_num_samples: int,
    cp: CheckpointManager,
    logger,
    dry_run: bool = False,
) -> dict[str, object]:
    classified = classify_tasks(tasks)
    completed_ids = set(cp.completed)
    metrics_parts, forecast_parts, naive_parts, completed_from_disk = _load_completed_task_outputs(run_root, classified)
    training_times: dict[str, float] = {}
    inference_times: dict[str, float] = {}
    completed_ids.update(completed_from_disk)
    cp.completed.update(completed_from_disk)
    plan = build_parallel_plan(classified, completed_ids)
    plan["completed_ids"] = sorted(completed_ids)
    plan["resource_detection"] = {
        "effective_cpu_cores": parallel_config.get("effective_cpu_cores"),
        "cpu_max": parallel_config.get("cpu_max"),
        "cpu_period": parallel_config.get("cpu_period"),
    }
    plan["parallel_config"] = parallel_config

    if dry_run:
        atomic_write_json(run_root / "dry_run_plan.json", plan)
        return {
            "plan": plan,
            "metrics": _concat(metrics_parts),
            "forecasts": _concat(forecast_parts),
            "naive": _concat(naive_parts),
            "completed_ids": completed_ids,
            "failed": [],
            "training_times": training_times,
            "inference_times": inference_times,
        }

    failures: list[dict[str, object]] = []
    pending_by_freq: dict[str, dict[str, deque[dict[str, object]]]] = {}
    for freq in sorted(data_by_frequency):
        freq_tasks = [task for task in classified if str(task["frequency"]) == freq and str(task["task_id"]) not in completed_ids]
        pending_by_freq[freq] = {
            "cpu": deque([task for task in freq_tasks if task_resource_class(task) == "cpu"]),
            "gpu": deque([task for task in freq_tasks if task_resource_class(task) == "gpu"]),
        }

    total_tasks = len(classified)
    completed_now = len(completed_ids)
    for freq in sorted(data_by_frequency):
        freq_tasks = [task for task in classified if str(task["frequency"]) == freq]
        if not freq_tasks:
            continue
        context = TaskWorkerContext(
            run_root=str(run_root),
            season_length=int(season_length_by_frequency[freq]),
            horizon=int(horizon_by_frequency[freq]),
            optuna_num_samples=int(optuna_num_samples),
            threads_per_cpu_task=int(parallel_config.get("forecast_threads_per_cpu_task", 1)),
            threads_per_gpu_task=int(parallel_config.get("forecast_threads_per_gpu_task", 2)),
            data_by_frequency={freq: data_by_frequency[freq]},
            cutoffs_by_frequency={freq: cutoffs_by_frequency[freq]},
            bucket_manifest=bucket_manifest,
        )

        cpu_total = int(parallel_config.get("forecast_cpu_task_workers", 1))
        gpu_total = int(parallel_config.get("forecast_gpu_task_workers", 1))
        stat_limit = int(parallel_config.get("forecast_statistical_workers", cpu_total))
        ml_limit = int(parallel_config.get("forecast_mlforecast_workers", cpu_total))
        neural_limit = int(parallel_config.get("forecast_neural_workers", gpu_total))
        transformer_limit = int(parallel_config.get("forecast_transformer_workers", gpu_total))

        cpu_pending = {
            "statistical": pending_by_freq[freq]["cpu"],
            "mlforecast": deque([task for task in pending_by_freq[freq]["cpu"] if str(task.get("model_family")) == "mlforecast"]),
        }
        cpu_pending["statistical"] = deque([task for task in pending_by_freq[freq]["cpu"] if str(task.get("model_family")) == "statistical"])
        gpu_pending = {
            "neural": deque([task for task in pending_by_freq[freq]["gpu"] if str(task.get("model_family")) == "neuralforecast"]),
            "transformer": deque([task for task in pending_by_freq[freq]["gpu"] if str(task.get("model_family")) == "transformers"]),
        }

        parallel_enabled = bool(parallel_config.get("forecast_parallel_enabled", False))

        # Always set up the main-process worker context (used for serial execution and
        # statistical threads, which share the parent-process global).
        initialize_worker(context)

        def _handle_result(task: dict[str, object], result: TaskExecutionResult) -> None:
            nonlocal completed_now
            if result.status == "done":
                cp.mark_done(_task_record(task, result))
                completed_ids.add(result.task_id)
                tdir = Path(result.task_dir)
                metrics_parts.append(_read_task_frame(tdir, "metrics.parquet"))
                forecast_parts.append(_read_task_frame(tdir, "forecasts.parquet"))
                naive_parts.append(_read_task_frame(tdir, "naive_forecasts.parquet"))
                training_times[result.task_id] = float(result.training_time_seconds or 0.0)
                inference_times[result.task_id] = float(result.inference_time_seconds or 0.0)
                completed_now += 1
                if completed_now % 3 == 0:
                    _write_merged_outputs(run_root, metrics_parts, forecast_parts, naive_parts)
            elif result.status == "already_done":
                completed_ids.add(result.task_id)
                done_payload = _read_done_payload(Path(result.task_dir))
                training_times[result.task_id] = float(done_payload.get("training_time_seconds", done_payload.get("training_seconds", result.training_time_seconds)) or 0.0)
                inference_times[result.task_id] = float(done_payload.get("inference_time_seconds", done_payload.get("inference_seconds", result.inference_time_seconds)) or 0.0)
                cp.mark_done(_task_record(task, done_payload))
                completed_now += 1
            else:
                cp.mark_failed(_task_record(task, result), RuntimeError(result.error or "task failed"))
                failures.append(asdict(result))
                logger.error("Task failed: %s", task["task_id"])

        def _run_pool(task_list: list[dict], executor: futures.Executor, fut_map: dict) -> None:
            """Submit tasks to executor; collect results in main thread (keeps checkpoint writes serial)."""
            if not task_list:
                return
            pending = {executor.submit(execute_task, t): t for t in task_list if str(t["task_id"]) not in completed_ids}
            for t in task_list:
                if str(t["task_id"]) in completed_ids:
                    done_payload = _read_done_payload(task_output_dir(run_root, str(t["task_id"])))
                    training_times[str(t["task_id"])] = float(done_payload.get("training_time_seconds", 0.0) or 0.0)
                    inference_times[str(t["task_id"])] = float(done_payload.get("inference_time_seconds", 0.0) or 0.0)
                    cp.mark_done(_task_record(t, done_payload))
            for fut in futures.as_completed(pending):
                task = pending[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    logger.error("Worker exception for task %s: %s", task.get("task_id"), exc)
                    # Mark as failed without crashing the whole run
                    from dataclasses import fields as _fields
                    fake = TaskExecutionResult(
                        task_id=str(task["task_id"]),
                        safe_task_id=str(task.get("safe_task_id", task["task_id"])),
                        task_dir=str(task_output_dir(run_root, str(task["task_id"]))),
                        resource_class=str(task.get("resource_class", "cpu")),
                        status="failed",
                        elapsed_seconds=0.0,
                        error=str(exc),
                    )
                    _handle_result(task, fake)
                    continue
                _handle_result(task, result)

        def _run_serial(task_list: list[dict]) -> None:
            for task in task_list:
                if str(task["task_id"]) in completed_ids:
                    done_payload = _read_done_payload(task_output_dir(run_root, str(task["task_id"])))
                    training_times[str(task["task_id"])] = float(done_payload.get("training_time_seconds", 0.0) or 0.0)
                    inference_times[str(task["task_id"])] = float(done_payload.get("inference_time_seconds", 0.0) or 0.0)
                    cp.mark_done(_task_record(task, done_payload))
                    continue
                result = execute_task(task)
                _handle_result(task, result)
                if completed_now % 3 == 0:
                    _write_merged_outputs(run_root, metrics_parts, forecast_parts, naive_parts)

        stat_tasks = list(cpu_pending["statistical"])
        ml_tasks = list(cpu_pending["mlforecast"])
        gpu_tasks = list(gpu_pending["neural"]) + list(gpu_pending["transformer"])

        if parallel_enabled:
            # Phase 1 — Statistical: ThreadPoolExecutor.
            #   statsforecast releases the GIL during its C/Fortran computations so threads
            #   achieve real parallelism. Forking statsforecast workers is unsafe (signal
            #   handlers in the Fortran ARIMA code can crash forked processes).
            if stat_tasks:
                with futures.ThreadPoolExecutor(max_workers=max(1, stat_limit)) as stat_ex:
                    _run_pool(stat_tasks, stat_ex, {})

            # Phase 2 — MLForecast: ProcessPoolExecutor with 'spawn' start method.
            #   Using 'spawn' instead of 'fork' avoids inheriting CUDA/PyTorch state from the
            #   parent process (initialized by prior GPU phases), which causes child crashes.
            if ml_tasks:
                _mp_ctx = multiprocessing.get_context("spawn")
                with futures.ProcessPoolExecutor(max_workers=max(1, ml_limit), mp_context=_mp_ctx, initializer=initialize_worker, initargs=(context,)) as ml_ex:
                    _run_pool(ml_tasks, ml_ex, {})

            # Phase 3 — GPU tasks: serial in main process.
            #   PyTorch/Lightning is incompatible with subprocess forking.
            _run_serial(gpu_tasks)
        else:
            _run_serial(stat_tasks + ml_tasks + gpu_tasks)

    metrics, forecasts, naive = _write_merged_outputs(run_root, metrics_parts, forecast_parts, naive_parts)
    cp.status(
        {
            "stage": "done",
            "running_tasks": [],
            "queued_tasks": 0,
            "resource_class_counts": plan["resource_class_counts"],
            "recent_completed_task_ids": sorted(completed_ids)[-10:],
            "recent_failed_task_ids": [item["task_id"] for item in failures[-10:]],
        },
        total_tasks,
    )
    return {
        "plan": plan,
        "metrics": metrics,
        "forecasts": forecasts,
        "naive": naive,
        "completed_ids": completed_ids,
        "failed": failures,
        "training_times": training_times,
        "inference_times": inference_times,
    }
