"""Tests for P0-003: checkpoint schema v2 and rich metadata columns."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.training.checkpointing import CHECKPOINT_SCHEMA_VERSION, COMPLETED_COLUMNS, CheckpointManager


def test_completed_parquet_has_v2_columns(tmp_path: Path):
    cp = CheckpointManager(tmp_path, "run_v2")
    record = {
        "task_id": "t1",
        "status": "done",
        "feature_name": "feature_non_normality",
        "frequency": "monthly",
        "model_family": "statistical",
        "model_name": "AutoARIMA",
        "decomposition_method": "without_stl",
        "finetuning_mode": "no_finetune",
        "started_at_utc": "2026-05-23T00:00:00Z",
        "finished_at_utc": "2026-05-23T00:00:01Z",
        "training_time_seconds": 1.5,
        "inference_time_seconds": 0.3,
        "output_dir": str(tmp_path),
        "fallback_reason": None,
        "model_output_source": "trained_statsforecast",
        "model_backend": "statsforecast",
        "backend_library_version": "1.9.0",
        "forecast_hash": "abc1234567890abc",
    }
    cp.mark_done(record)

    completed = pd.read_parquet(tmp_path / "checkpoints" / "completed_tasks.parquet")
    v2_required = {"model_backend", "backend_library_version", "forecast_hash"}
    assert v2_required.issubset(set(completed.columns)), f"Missing v2 columns: {v2_required - set(completed.columns)}"
    assert set(COMPLETED_COLUMNS).issubset(set(completed.columns))


def test_schema_version_in_manifest(tmp_path: Path):
    cp = CheckpointManager(tmp_path, "run_manifest")
    cp.mark_done(
        {
            "task_id": "t2",
            "status": "done",
            "started_at_utc": "2026-05-23T00:00:00Z",
            "finished_at_utc": "2026-05-23T00:00:01Z",
            "training_time_seconds": 0.5,
            "inference_time_seconds": 0.1,
        }
    )
    manifest = json.loads((tmp_path / "checkpoints" / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert manifest["schema_version"] == 2


def test_checkpoint_schema_version_constant():
    assert CHECKPOINT_SCHEMA_VERSION == 2


def test_completed_columns_includes_all_v2_fields():
    required = {"model_backend", "backend_library_version", "forecast_hash"}
    assert required.issubset(set(COMPLETED_COLUMNS))
