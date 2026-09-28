"""Tests for P0-001/P0-002: strict fallback/provenance schema and statistical mislabeling."""
from __future__ import annotations

import os
import importlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.training.strict_mode import StrictModeViolation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_strict(monkeypatch, value: str) -> None:
    monkeypatch.setenv("STRICT_MODEL_MODE", value)
    # Force re-evaluation on each call to is_strict_mode().


def _tiny_train() -> np.ndarray:
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


# ---------------------------------------------------------------------------
# statistical.py — import error path
# ---------------------------------------------------------------------------

def test_statistical_nonstrict_import_error_labeled(monkeypatch):
    _set_strict(monkeypatch, "0")
    with patch.dict("sys.modules", {"statsforecast": None, "statsforecast.models": None}):
        import src.models.statistical as stat_mod
        importlib.reload(stat_mod)
        result = stat_mod.forecast_statistical("AutoARIMA", _tiny_train(), h=2, season_length=3)
    assert result.model_output_source == "deterministic_baseline_fallback"
    assert result.fit_status == "fallback"
    assert result.fallback_reason and "import_error" in result.fallback_reason


def test_statistical_strict_import_error_raises(monkeypatch):
    _set_strict(monkeypatch, "1")
    with patch.dict("sys.modules", {"statsforecast": None, "statsforecast.models": None}):
        import src.models.statistical as stat_mod
        importlib.reload(stat_mod)
        with pytest.raises(StrictModeViolation, match="statsforecast"):
            stat_mod.forecast_statistical("AutoARIMA", _tiny_train(), h=2, season_length=3)


# ---------------------------------------------------------------------------
# statistical.py — fit failure path
# ---------------------------------------------------------------------------

def test_statistical_nonstrict_fit_failure_labeled(monkeypatch):
    _set_strict(monkeypatch, "0")
    import src.models.statistical as stat_mod
    importlib.reload(stat_mod)

    # StatsForecast is imported inside the function body, so patch at the source module.
    mock_sf = MagicMock()
    mock_sf.return_value.forecast.side_effect = RuntimeError("boom")
    with patch("statsforecast.StatsForecast", mock_sf):
        result = stat_mod.forecast_statistical("AutoARIMA", _tiny_train(), h=2, season_length=3)

    assert result.model_output_source == "deterministic_baseline_fallback"
    assert result.fit_status == "fallback"
    assert "fit_failed" in (result.fallback_reason or "")


def test_statistical_strict_fit_failure_raises(monkeypatch):
    _set_strict(monkeypatch, "1")
    import src.models.statistical as stat_mod
    importlib.reload(stat_mod)

    mock_sf = MagicMock()
    mock_sf.return_value.forecast.side_effect = RuntimeError("boom")
    with patch("statsforecast.StatsForecast", mock_sf):
        with pytest.raises(StrictModeViolation, match="fit failed"):
            stat_mod.forecast_statistical("AutoARIMA", _tiny_train(), h=2, season_length=3)


# ---------------------------------------------------------------------------
# AutoSARIMA
# ---------------------------------------------------------------------------

def test_autosarima_strict_trains_as_real_model(monkeypatch):
    # AutoSARIMA is now a real model (D=1 forced seasonal differencing), not an alias.
    # It must train successfully in strict mode without raising StrictModeViolation.
    _set_strict(monkeypatch, "1")
    import src.models.statistical as stat_mod
    importlib.reload(stat_mod)
    y = np.array([float(i % 12 + 1) for i in range(36)])
    result = stat_mod.forecast_statistical("AutoSARIMA", y, h=6, season_length=12)
    assert result.model_output_source == "trained_statsforecast"
    assert result.fit_status == "trained"


def test_autosarima_nonstrict_returns_trained_result(monkeypatch):
    # AutoSARIMA now trains a genuine SARIMA (D=1); it should not produce a fallback.
    _set_strict(monkeypatch, "0")
    import src.models.statistical as stat_mod
    importlib.reload(stat_mod)
    y = np.array([float(i % 12 + 1) for i in range(36)])
    result = stat_mod.forecast_statistical("AutoSARIMA", y, h=6, season_length=12)
    assert result.model_output_source == "trained_statsforecast"
    assert result.fallback_reason is None


# ---------------------------------------------------------------------------
# mlforecast, neural, transformer — strict rejection
# ---------------------------------------------------------------------------

def test_mlforecast_strict_rejected(monkeypatch):
    _set_strict(monkeypatch, "1")
    import src.models.mlforecast_auto as mod
    importlib.reload(mod)
    with pytest.raises(StrictModeViolation, match="MLForecast"):
        mod.forecast_mlforecast("AutoRidge", _tiny_train(), h=2, season_length=3)


def test_mlforecast_nonstrict_fallback_labeled(monkeypatch):
    _set_strict(monkeypatch, "0")
    import src.models.mlforecast_auto as mod
    importlib.reload(mod)
    result = mod.forecast_mlforecast("AutoRidge", _tiny_train(), h=2, season_length=3)
    assert result.model_output_source == "deterministic_baseline_fallback"
    assert result.fit_status == "fallback"
    assert result.fallback_reason == "mlforecast_not_implemented"


def test_neural_strict_rejected(monkeypatch):
    _set_strict(monkeypatch, "1")
    import src.models.neural_auto as mod
    importlib.reload(mod)
    with pytest.raises(StrictModeViolation, match="NeuralForecast"):
        mod.forecast_neural("AutoNLinear", _tiny_train(), h=2, season_length=3)


def test_transformer_strict_rejected(monkeypatch):
    _set_strict(monkeypatch, "1")
    import src.models.transformer_auto as mod
    importlib.reload(mod)
    with pytest.raises(StrictModeViolation, match="Transformer"):
        mod.forecast_transformer("AutoTFT", _tiny_train(), h=2, season_length=3)


def test_component_strict_rejected(monkeypatch):
    _set_strict(monkeypatch, "1")
    import src.models.component_wrappers as mod
    importlib.reload(mod)
    with pytest.raises(StrictModeViolation, match="[Cc]omponent"):
        mod.forecast_component("AutoARIMA", _tiny_train(), h=2, season_length=3)


# ---------------------------------------------------------------------------
# All metrics rows must carry provenance columns
# ---------------------------------------------------------------------------

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
    bucket_manifest = pd.DataFrame({"feature_name": ["feature_non_normality"], "unique_id": ["s1"], "feature_tercile_bucket": ["Low"]})
    return frame, cutoffs, bucket_manifest


def test_all_metrics_rows_have_provenance_columns(monkeypatch):
    _set_strict(monkeypatch, "0")
    from src.training.train_family import evaluate_task
    df, cutoffs, bm = _make_eval_inputs()
    task = {
        "task_id": "stat|prov",
        "feature_name": "feature_non_normality",
        "frequency": "monthly",
        "decomposition_method": "without_stl",
        "model_family": "statistical",
        "model_name": "AutoARIMA",
        "finetuning_mode": "no_finetune",
    }
    result = evaluate_task(task, df, cutoffs, season_length=12, horizon=1, bucket_manifest=bm)
    required = {"model_output_source", "fallback_reason", "model_backend", "backend_library_version", "forecast_hash"}
    assert required.issubset(set(result.metrics.columns)), f"Missing columns: {required - set(result.metrics.columns)}"
    assert result.forecast_hash is not None
    assert len(result.forecast_hash) == 16  # SHA256[:16]


def test_nonstrict_fallback_result_has_valid_hash(monkeypatch):
    _set_strict(monkeypatch, "0")
    import src.models.mlforecast_auto as mod
    importlib.reload(mod)
    result = mod.forecast_mlforecast("AutoRidge", _tiny_train(), h=3, season_length=3)
    assert result.forecast_hash is not None
    assert len(result.forecast_hash) == 16
    assert len(result.yhat) == 3
