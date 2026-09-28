from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from src.training.parallel_runtime import TaskWorkerContext, execute_task, initialize_worker
from src.training.strict_mode import StrictModeViolation


def _skip_missing(package: str) -> None:
    if importlib.util.find_spec(package) is None:
        pytest.skip(f"{package} is not installed in this environment")


def _global_frame(n_series: int = 3, n: int = 24, freq: str = "MS") -> pd.DataFrame:
    rng = np.random.default_rng(20260524)
    rows = []
    for i in range(n_series):
        uid = f"s{i}"
        dates = pd.date_range("2020-01-01", periods=n, freq=freq)
        y = 10 + i + np.sin(np.arange(n) / 3.0) + rng.normal(0, 0.05, n)
        for ds, value in zip(dates, y):
            rows.append({"unique_id": uid, "ds": ds, "y": float(value), "source_dataset": "synthetic"})
    return pd.DataFrame(rows)


def _assert_real_results(results: dict, expected_source: str, h: int) -> None:
    assert results
    for result in results.values():
        assert result.model_output_source == expected_source
        assert result.fit_status == "trained"
        assert result.fallback_reason is None
        assert len(result.yhat) == h
        assert np.isfinite(result.yhat).all()


def test_tiny_mlforecast_real_fit_predict() -> None:
    _skip_missing("mlforecast")
    from src.models.mlforecast_auto import forecast_mlforecast_global

    results = forecast_mlforecast_global("AutoRidge", _global_frame(), h=2, season_length=4, freq_str="MS", num_optuna_samples=1)
    _assert_real_results(results, "trained_mlforecast", h=2)


def test_tiny_neuralforecast_real_fit_predict() -> None:
    _skip_missing("neuralforecast")
    from src.models.neural_auto import forecast_neural_global

    results = forecast_neural_global("AutoNLinear", _global_frame(), h=2, season_length=4, freq_str="MS", num_optuna_samples=1)
    _assert_real_results(results, "trained_neuralforecast", h=2)


def test_tiny_transformer_real_fit_predict() -> None:
    _skip_missing("neuralforecast")
    from src.models.transformer_auto import forecast_transformer_global

    results = forecast_transformer_global("AutoPatchTST", _global_frame(), h=2, season_length=4, freq_str="MS", num_optuna_samples=1)
    _assert_real_results(results, "trained_transformer", h=2)


def test_strict_worker_failure_writes_no_fallback_rows(monkeypatch, tmp_path) -> None:
    import src.training.parallel_runtime as runtime

    monkeypatch.setenv("STRICT_MODEL_MODE", "1")

    def fail_evaluate(*args, **kwargs):
        raise StrictModeViolation("real backend failure")

    monkeypatch.setattr(runtime, "evaluate_task", fail_evaluate)
    task = {
        "task_id": "feat|monthly|without_stl|mlforecast|AutoRidge|no_finetune",
        "feature_name": "feat",
        "frequency": "monthly",
        "decomposition_method": "without_stl",
        "model_family": "mlforecast",
        "model_name": "AutoRidge",
        "finetuning_mode": "no_finetune",
    }
    df = _global_frame(n_series=1, n=8)
    context = TaskWorkerContext(
        run_root=str(tmp_path),
        season_length=4,
        horizon=2,
        optuna_num_samples=1,
        threads_per_cpu_task=1,
        threads_per_gpu_task=1,
        data_by_frequency={"monthly": df},
        cutoffs_by_frequency={"monthly": pd.DataFrame()},
        bucket_manifest=pd.DataFrame(),
    )
    initialize_worker(context)

    result = execute_task(task)
    task_dir = tmp_path / "task_outputs" / "feat_monthly_without_stl_mlforecast_AutoRidge_no_finetune"

    assert result.status == "failed"
    assert result.model_output_source == "strict_mode_failure"
    assert (task_dir / "FAILED.json").exists()
    assert not (task_dir / "DONE.json").exists()
    assert not (task_dir / "metrics.parquet").exists()
