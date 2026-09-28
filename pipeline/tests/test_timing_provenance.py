"""Tests for P0-003: timing propagation and strict leaderboard validation."""
from __future__ import annotations

import pandas as pd
import pytest

from src.training.strict_mode import StrictModeViolation


def _make_metrics_df(model_output_source: str = "trained_statsforecast") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_name": "feature_non_normality",
                "frequency": "monthly",
                "model_family": "statistical",
                "model_name": "AutoARIMA",
                "decomposition_method": "without_stl",
                "finetuning_mode": "no_finetune",
                "rel_naive_unclipped": 0.8,
                "rel_naive_clipped": 0.8,
                "mae": 1.0,
                "smape": 0.1,
                "mase": 0.9,
                "wape": 0.1,
                "unique_id": "s1",
                "window": 0,
                "model_output_source": model_output_source,
            }
        ]
    )


def _make_eval_inputs():
    frame = pd.DataFrame(
        {
            "unique_id": ["s1"] * 6,
            "ds": pd.date_range("2020-01-01", periods=6, freq="ME"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "source_dataset": ["M3_Monthly"] * 6,
        }
    )
    cutoffs = pd.DataFrame({"unique_id": ["s1"], "train_end_idx": [4], "window": [0], "cutoff": [4]})
    bucket_manifest = pd.DataFrame(
        {"feature_name": ["feature_non_normality"], "unique_id": ["s1"], "feature_tercile_bucket": ["Low"]}
    )
    return frame, cutoffs, bucket_manifest


def test_evaluate_task_emits_timings(monkeypatch):
    monkeypatch.delenv("STRICT_MODEL_MODE", raising=False)
    from src.training.train_family import evaluate_task

    df, cutoffs, bm = _make_eval_inputs()
    task = {
        "task_id": "stat|timing",
        "feature_name": "feature_non_normality",
        "frequency": "monthly",
        "decomposition_method": "without_stl",
        "model_family": "statistical",
        "model_name": "AutoARIMA",
        "finetuning_mode": "no_finetune",
    }
    result = evaluate_task(task, df, cutoffs, season_length=12, horizon=1, bucket_manifest=bm)
    assert result.training_time_seconds > 0
    assert result.inference_time_seconds > 0
    assert result.started_at_utc
    assert result.finished_at_utc
    assert {"training_time_seconds", "inference_time_seconds"}.issubset(result.metrics.columns)


def test_leaderboard_strict_rejects_zero_timing(monkeypatch):
    monkeypatch.setenv("STRICT_MODEL_MODE", "1")
    from src.evaluation.leaderboards import overall_leaderboard

    metrics = _make_metrics_df("trained_statsforecast")
    task_id = "feature_non_normality|monthly|statistical|AutoARIMA|without_stl|no_finetune"
    with pytest.raises(StrictModeViolation, match="training_time_seconds"):
        overall_leaderboard(metrics, {task_id: 0.0}, {task_id: 0.5}, num_samples=1, strict_mode=True)


def test_leaderboard_strict_rejects_missing_timing(monkeypatch):
    monkeypatch.setenv("STRICT_MODEL_MODE", "1")
    from src.evaluation.leaderboards import overall_leaderboard

    metrics = _make_metrics_df("trained_statsforecast")
    with pytest.raises(StrictModeViolation, match="training_time_seconds"):
        overall_leaderboard(metrics, {}, {}, num_samples=1, strict_mode=True)


def test_leaderboard_nonstrict_zero_timing_allowed(monkeypatch):
    monkeypatch.delenv("STRICT_MODEL_MODE", raising=False)
    from src.evaluation.leaderboards import overall_leaderboard

    metrics = _make_metrics_df("deterministic_baseline_fallback")
    result = overall_leaderboard(metrics, {}, {}, num_samples=1, strict_mode=False)
    assert not result.empty
    assert result["training_time_seconds"].iloc[0] == 0.0


def test_leaderboard_benchmark_valid_flag(monkeypatch):
    monkeypatch.delenv("STRICT_MODEL_MODE", raising=False)
    from src.evaluation.leaderboards import overall_leaderboard

    task_id = "feature_non_normality|monthly|statistical|AutoARIMA|without_stl|no_finetune"
    trained_metrics = _make_metrics_df("trained_statsforecast")
    result = overall_leaderboard(trained_metrics, {task_id: 1.5}, {task_id: 0.2}, num_samples=1, strict_mode=False)
    assert bool(result["benchmark_valid"].iloc[0]) is True

    fallback_metrics = _make_metrics_df("deterministic_baseline_fallback")
    result2 = overall_leaderboard(fallback_metrics, {task_id: 1.5}, {task_id: 0.2}, num_samples=1, strict_mode=False)
    assert bool(result2["benchmark_valid"].iloc[0]) is False
