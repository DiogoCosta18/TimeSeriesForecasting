"""Tests for P0-004: honest Optuna/tuning status (no placeholder_or_resumed)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.training.strict_mode import StrictModeViolation
from src.training.tune import write_optuna_artifacts, write_placeholder_optuna_artifacts


def test_disabled_by_budget_writes_honest_status(tmp_path: Path):
    (tmp_path / "optuna").mkdir()
    write_optuna_artifacts(tmp_path, "AutoARIMA", num_samples=2, tuning_status="disabled_by_budget")
    payload = json.loads((tmp_path / "optuna" / "best_params_AutoARIMA.json").read_text(encoding="utf-8"))
    assert payload["tuning_status"] == "disabled_by_budget"
    assert "placeholder_or_resumed" not in str(payload)


def test_disabled_by_budget_writes_trials(tmp_path: Path):
    (tmp_path / "optuna").mkdir()
    write_optuna_artifacts(tmp_path, "AutoETS", num_samples=5, tuning_status="disabled_by_budget")
    trials = pd.read_parquet(tmp_path / "optuna" / "trials_AutoETS.parquet")
    assert "tuning_status" in trials.columns
    assert trials["tuning_status"].iloc[0] == "disabled_by_budget"
    assert "placeholder_or_resumed" not in trials["tuning_status"].iloc[0]


def test_placeholder_status_strict_rejected(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("STRICT_MODEL_MODE", "1")
    (tmp_path / "optuna").mkdir()
    with pytest.raises(StrictModeViolation, match="placeholder"):
        write_optuna_artifacts(tmp_path, "AutoARIMA", num_samples=2, tuning_status="placeholder")


def test_placeholder_status_nonstrict_allowed(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("STRICT_MODEL_MODE", raising=False)
    (tmp_path / "optuna").mkdir()
    # Should not raise in non-strict mode
    write_optuna_artifacts(tmp_path, "AutoARIMA", num_samples=2, tuning_status="placeholder")
    payload = json.loads((tmp_path / "optuna" / "best_params_AutoARIMA.json").read_text(encoding="utf-8"))
    assert payload["tuning_status"] == "placeholder"


def test_backwards_compat_write_placeholder_uses_disabled_by_budget(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("STRICT_MODEL_MODE", raising=False)
    (tmp_path / "optuna").mkdir()
    # Legacy function now uses disabled_by_budget, not placeholder_or_resumed
    write_placeholder_optuna_artifacts(tmp_path, "AutoRidge", num_samples=3)
    payload = json.loads((tmp_path / "optuna" / "best_params_AutoRidge.json").read_text(encoding="utf-8"))
    assert payload["tuning_status"] == "disabled_by_budget"
    assert "placeholder_or_resumed" not in str(payload)


def test_parallel_runtime_writes_disabled_by_budget(monkeypatch, tmp_path: Path):
    """execute_task writes disabled_by_budget in the optuna dir, not placeholder_or_resumed."""
    monkeypatch.delenv("STRICT_MODEL_MODE", raising=False)
    import numpy as np
    import pandas as pd

    from src.training.parallel_runtime import TaskWorkerContext, execute_task, initialize_worker

    frame = pd.DataFrame(
        {
            "unique_id": ["s1"] * 6,
            "ds": pd.date_range("2020-01-01", periods=6, freq="ME"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "source_dataset": ["M3_Monthly"] * 6,
        }
    )
    cutoffs = pd.DataFrame({"unique_id": ["s1"], "train_end_idx": [4], "window": [0], "cutoff": [4]})
    bm = pd.DataFrame({"feature_name": ["feature_non_normality"], "unique_id": ["s1"], "feature_tercile_bucket": ["Low"]})
    ctx = TaskWorkerContext(
        run_root=str(tmp_path),
        season_length=12,
        horizon=1,
        optuna_num_samples=2,
        threads_per_cpu_task=1,
        threads_per_gpu_task=2,
        data_by_frequency={"monthly": frame},
        cutoffs_by_frequency={"monthly": cutoffs},
        bucket_manifest=bm,
    )
    initialize_worker(ctx)
    task = {
        "task_id": "stat_optuna_test",
        "feature_name": "feature_non_normality",
        "frequency": "monthly",
        "decomposition_method": "without_stl",
        "model_family": "statistical",
        "model_name": "AutoARIMA",
        "finetuning_mode": "no_finetune",
    }
    from src.training.resources import task_output_dir
    tdir = task_output_dir(tmp_path, "stat_optuna_test")
    result = execute_task(task)
    assert result.status == "done"

    params_path = tdir / "optuna" / "best_params_AutoARIMA.json"
    assert params_path.exists()
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    assert payload["tuning_status"] == "disabled_by_budget"
    assert "placeholder_or_resumed" not in str(payload)
