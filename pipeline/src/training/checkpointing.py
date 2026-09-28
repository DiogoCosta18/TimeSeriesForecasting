from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from src.training.gpu_utils import gpu_memory
from src.utils import atomic_write_json, utc_now_iso, write_dataframe


TERMINAL_COMPLETE_STATUSES = {"done", "already_done", "skipped"}
CHECKPOINT_SCHEMA_VERSION = 2
COMPLETED_COLUMNS = [
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
    "model_output_source",
    "model_backend",
    "backend_library_version",
    "forecast_hash",
]


class CheckpointManager:
    def __init__(self, run_root: Path, run_id: str):
        self.run_root = run_root
        self.run_id = run_id
        self.completed_path = run_root / "checkpoints" / "completed_tasks.parquet"
        self.manifest_path = run_root / "checkpoints" / "checkpoint_manifest.json"
        self.latest_path = run_root / "checkpoints" / "latest_state.json"
        self.status_path = run_root / "STATUS.json"
        self.lock_path = run_root / "checkpoints" / "checkpoint.lock"
        self.started = time.time()
        self.completed: set[str] = set()
        self.completed_records: dict[str, dict[str, Any]] = {}
        self.failed: list[dict[str, Any]] = []
        self.load()

    @contextmanager
    def _locked(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = None
        while handle is None:
            try:
                handle = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            except FileExistsError:
                time.sleep(0.05)
        try:
            yield
        finally:
            try:
                os.close(handle)
            finally:
                try:
                    self.lock_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def load(self) -> None:
        if self.completed_path.exists():
            try:
                df = pd.read_parquet(self.completed_path)
                if not df.empty and "task_id" in df.columns:
                    for _, row in df.iterrows():
                        record = row.to_dict()
                        task_id = str(record.get("task_id", "")).strip()
                        if not task_id:
                            continue
                        self.completed_records[task_id] = record
                        if str(record.get("status", "done")) in TERMINAL_COMPLETE_STATUSES:
                            self.completed.add(task_id)
            except Exception:
                csv = self.completed_path.with_suffix(".csv")
                if csv.exists():
                    df = pd.read_csv(csv)
                    if not df.empty and "task_id" in df.columns:
                        self.completed = set(df["task_id"].astype(str))
                        for _, row in df.iterrows():
                            record = row.to_dict()
                            task_id = str(record.get("task_id", "")).strip()
                            if task_id:
                                self.completed_records[task_id] = record

    def is_done(self, task_id: str) -> bool:
        return task_id in self.completed

    def _write_completed_records(self) -> None:
        if self.completed_records:
            df = pd.DataFrame(list(self.completed_records.values()))
        else:
            df = pd.DataFrame(columns=COMPLETED_COLUMNS)
        for column in COMPLETED_COLUMNS:
            if column not in df.columns:
                df[column] = None
        ordered = [column for column in COMPLETED_COLUMNS if column in df.columns]
        ordered.extend([column for column in df.columns if column not in ordered])
        df = df[ordered]
        if not df.empty:
            df = df.sort_values(["task_id"], na_position="last")
        write_dataframe(self.completed_path, df)

    def mark_done(self, record: dict[str, Any]) -> None:
        task_id = str(record["task_id"])
        with self._locked():
            normalized = dict(record)
            normalized.setdefault("status", "done")
            normalized.setdefault("finished_at_utc", utc_now_iso())
            self.completed_records[task_id] = normalized
            self.completed.add(task_id)
            self._write_completed_records()
            atomic_write_json(self.manifest_path, {"run_id": self.run_id, "completed_tasks": len(self.completed), "updated_at_utc": utc_now_iso(), "schema_version": CHECKPOINT_SCHEMA_VERSION})

    def mark_failed(self, record: dict[str, Any], error: Exception) -> None:
        task_id = str(record.get("task_id", "unknown"))
        with self._locked():
            failed_record = dict(record)
            failed_record.setdefault("task_id", task_id)
            failed_record.setdefault("status", "failed")
            failed_record["error"] = repr(error)
            failed_record.setdefault("failed_at_utc", utc_now_iso())
            self.completed_records[task_id] = failed_record
            self.failed.append(failed_record)
            self._write_completed_records()

    def status(self, current: dict[str, Any], total_tasks: int) -> None:
        with self._locked():
            mem = gpu_memory()
            data = {
                "run_id": self.run_id,
                "current_stage": current.get("stage", "running"),
                "current_feature": current.get("feature_name"),
                "current_frequency": current.get("frequency"),
                "current_bucket": current.get("feature_bucket"),
                "current_decomposition_method": current.get("decomposition_method"),
                "current_model_family": current.get("model_family"),
                "current_model_name": current.get("model_name"),
                "current_finetuning_mode": current.get("finetuning_mode"),
                    "current_model_output_source": current.get("model_output_source"),
                    "current_fallback_reason": current.get("fallback_reason"),
                "running_tasks": current.get("running_tasks", []),
                "queued_tasks": current.get("queued_tasks", 0),
                "completed_tasks": len(self.completed),
                "failed_tasks": len(self.failed),
                "recent_completed_task_ids": current.get("recent_completed_task_ids", []),
                "recent_failed_task_ids": current.get("recent_failed_task_ids", []),
                "resource_class_counts": current.get("resource_class_counts", {}),
                "parallel_enabled": current.get("parallel_enabled", False),
                "effective_cpu_cores": current.get("effective_cpu_cores"),
                "cpu_task_workers": current.get("cpu_task_workers"),
                "gpu_task_workers": current.get("gpu_task_workers"),
                "last_checkpoint_time_utc": utc_now_iso(),
                "process_pid": os.getpid(),
                "elapsed_seconds": round(time.time() - self.started, 1),
                "estimated_remaining_seconds": None,
                **mem,
            }
            for key, value in current.items():
                if key not in data and key != "stage":
                    data[key] = value
            atomic_write_json(self.status_path, data)
            atomic_write_json(self.latest_path, data)
            status_log = self.run_root / "logs" / "status.log"
            status_log.parent.mkdir(parents=True, exist_ok=True)
            with status_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, sort_keys=True, default=str) + "\n")
