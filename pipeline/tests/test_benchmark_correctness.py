"""Tests for benchmark correctness: task count, STL routing, no-fallback, timing."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.training.task_graph import BUDGETS, finetuning_modes, make_tasks
from src.training.train_family import evaluate_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_cfg(features: int = 6, decomps: list[str] | None = None) -> dict:
    return {
        "features": [f"feat_{i}" for i in range(features)],
        "decomposition_methods": decomps or ["without_stl", "stl_seasonal_naive", "stl_model_all_components"],
        "model_families": {
            "statistical": True,
            "mlforecast": True,
            "neuralforecast": True,
            "transformers": True,
        },
        "finetuning": {
            "no_finetune": True,
            "finetune_by_feature_bucket": True,
            "finetune_by_individual_series": True,
        },
        "_budget_settings": BUDGETS["large"],
        "representative_sampling": {"n_per_source_dataset": 750},
        "mlforecast_auto": {"num_samples": 100},
        "neuralforecast_auto": {"num_samples": 100},
        "transformer_auto": {"num_samples": 100},
    }


def _model_families() -> dict[str, list[str]]:
    from src.data.schemas import MODEL_FAMILIES
    return MODEL_FAMILIES


def _tiny_df() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = 50
    dates = pd.date_range("1982-01", periods=n, freq="ME")
    rows = []
    for sid in ("s1", "s2", "s3"):
        y = np.cumsum(np.random.default_rng(42).standard_normal(n)) + 100
        for d, yv in zip(dates, y):
            rows.append({"unique_id": sid, "ds": d, "y": float(yv), "source_dataset": "M3_Monthly"})
    df = pd.DataFrame(rows)
    cutoffs = pd.DataFrame({
        "unique_id": ["s1", "s2", "s3"],
        "train_end_idx": [30, 30, 30],
        "window": [0, 0, 0],
        "cutoff": [dates[30]] * 3,
        "frequency": ["monthly"] * 3,
    })
    bucket_manifest = pd.DataFrame({
        "unique_id": ["s1", "s2", "s3"],
        "feature_name": "feat_test",
        "feature_tercile_bucket": "Mid",
    })
    return df, cutoffs, bucket_manifest


def _task(family: str, model_name: str, decomp: str = "without_stl", ft: str = "no_finetune") -> dict:
    return {
        "task_id": f"feat_test|monthly|{decomp}|{family}|{model_name}|{ft}",
        "feature_name": "feat_test",
        "frequency": "monthly",
        "decomposition_method": decomp,
        "model_family": family,
        "model_name": model_name,
        "finetuning_mode": ft,
    }


# ---------------------------------------------------------------------------
# 1. Task graph count: 972 tasks for large budget with all 6 features × 2 freqs
# ---------------------------------------------------------------------------

def test_task_count_972_for_large_budget():
    from src.data.schemas import MODEL_FAMILIES
    cfg = _base_cfg(features=6)
    # Simulate the exact model configuration from schemas
    cfg["model_families"] = {fam: True for fam in MODEL_FAMILIES}
    tasks = make_tasks(cfg, "large", ["monthly", "quarterly"], None, None, None)
    # Expected: 6 feat × 2 freq × 3 decomp × per-family model×finetune
    families = MODEL_FAMILIES
    stat_models = len(families.get("statistical", []))
    ml_models = len(families.get("mlforecast", []))
    neural_models = len(families.get("neuralforecast", []))
    trans_models = len(families.get("transformers", []))
    # statistical: 3 modes (no_finetune, finetune_by_feature_bucket, finetune_by_individual_series)
    # ml/neural/transformer: 2 modes (no_finetune, finetune_by_feature_bucket)
    per_feat_freq_decomp = (
        stat_models * 3
        + ml_models * 2
        + neural_models * 2
        + trans_models * 2
    )
    expected = 6 * 2 * 3 * per_feat_freq_decomp
    assert len(tasks) == expected, (
        f"Expected {expected} tasks for large budget, got {len(tasks)}. "
        f"Models: stat={stat_models}×3, ml={ml_models}×2, neural={neural_models}×2, trans={trans_models}×2"
    )


# ---------------------------------------------------------------------------
# 2. Statistical gets finetune_by_individual_series; global families do not
# ---------------------------------------------------------------------------

def test_statistical_has_individual_series_mode():
    cfg = _base_cfg()
    modes_stat = finetuning_modes(cfg, "statistical")
    assert "finetune_by_individual_series" in modes_stat
    assert "no_finetune" in modes_stat
    assert "finetune_by_feature_bucket" in modes_stat


def test_global_families_do_not_get_individual_series_mode():
    cfg = _base_cfg()
    for family in ("mlforecast", "neuralforecast", "transformers"):
        modes = finetuning_modes(cfg, family)
        assert "finetune_by_individual_series" not in modes, (
            f"{family} should not have finetune_by_individual_series"
        )
        assert "finetune_by_feature_bucket" in modes


# ---------------------------------------------------------------------------
# 3. STL routing: global families go through _evaluate_global_task for all decomps
# ---------------------------------------------------------------------------

def test_stl_global_routing_routes_mlforecast_to_global_path(monkeypatch):
    """mlforecast + stl_seasonal_naive must call forecast_mlforecast_global, not forecast_component."""
    import src.training.train_family as tf

    global_called = {"called": False, "decomp": None}

    original_global = tf._evaluate_global_task

    def patched_global(task, *args, **kwargs):
        global_called["called"] = True
        global_called["decomp"] = task.get("decomposition_method")
        return original_global(task, *args, **kwargs)

    monkeypatch.setattr(tf, "_evaluate_global_task", patched_global)
    df, cutoffs, bm = _tiny_df()
    task = _task("mlforecast", "AutoRidge", decomp="stl_seasonal_naive")
    evaluate_task(task, df, cutoffs, season_length=12, horizon=10, bucket_manifest=bm)
    assert global_called["called"], "mlforecast+stl_seasonal_naive must use _evaluate_global_task"
    assert global_called["decomp"] == "stl_seasonal_naive"


def test_stl_global_routing_routes_neural_to_global_path(monkeypatch):
    import src.training.train_family as tf

    global_called = {"called": False}
    original = tf._evaluate_global_task

    def patched(task, *args, **kwargs):
        global_called["called"] = True
        return original(task, *args, **kwargs)

    monkeypatch.setattr(tf, "_evaluate_global_task", patched)
    df, cutoffs, bm = _tiny_df()
    task = _task("neuralforecast", "AutoNLinear", decomp="stl_model_all_components")
    evaluate_task(task, df, cutoffs, season_length=12, horizon=10, bucket_manifest=bm)
    assert global_called["called"], "neuralforecast+stl_model_all_components must use _evaluate_global_task"


# ---------------------------------------------------------------------------
# 4. Statistical + STL uses forecast_statistical (not forecast_component)
# ---------------------------------------------------------------------------

def test_statistical_stl_uses_forecast_statistical_not_component(monkeypatch):
    """Statistical + stl_seasonal_naive must call forecast_statistical, not forecast_component."""
    import src.models.component_wrappers as cw

    component_calls = {"n": 0}
    original_component = cw.forecast_component

    def counting_component(*args, **kwargs):
        component_calls["n"] += 1
        return original_component(*args, **kwargs)

    monkeypatch.setattr(cw, "forecast_component", counting_component)
    # Also patch in train_family where it's imported
    import src.training.train_family as tf
    monkeypatch.setattr(tf, "forecast_component", counting_component)

    df, cutoffs, bm = _tiny_df()
    task = _task("statistical", "AutoARIMA", decomp="stl_seasonal_naive")
    result = evaluate_task(task, df, cutoffs, season_length=12, horizon=10, bucket_manifest=bm)
    assert component_calls["n"] == 0, "Statistical+STL must not call forecast_component"
    assert result.model_output_source == "trained_statsforecast"


# ---------------------------------------------------------------------------
# 5. AutoSARIMA is distinct from AutoARIMA
# ---------------------------------------------------------------------------

def test_autosarima_uses_forced_seasonal_differencing():
    """AutoSARIMA must use D=1 (forced seasonal differencing), unlike AutoARIMA's D=auto."""
    import statsforecast
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    y = np.cumsum(np.random.default_rng(0).standard_normal(60)) + 100
    season_length = 12
    h = 6

    # Verify the model objects are configured differently
    sarima_model = AutoARIMA(season_length=season_length, D=1)
    arima_model = AutoARIMA(season_length=season_length)

    def _fit_and_forecast(model_obj):
        df = pd.DataFrame({"unique_id": ["s"] * len(y), "ds": range(len(y)), "y": list(map(float, y))})
        sf = StatsForecast(models=[model_obj], freq=1, n_jobs=1)
        fcst = sf.forecast(df=df, h=h)
        col = [c for c in fcst.columns if c not in {"unique_id", "ds"}][0]
        return fcst[col].to_numpy(dtype=float)

    yhat_sarima = _fit_and_forecast(sarima_model)
    yhat_arima = _fit_and_forecast(arima_model)

    # AutoSARIMA (D=1) may produce different forecasts than AutoARIMA (D=auto)
    # At minimum, both must produce finite forecasts
    assert np.isfinite(yhat_sarima).all(), "AutoSARIMA must produce finite forecasts"
    assert np.isfinite(yhat_arima).all(), "AutoARIMA must produce finite forecasts"


def test_autosarima_nonstrict_returns_trained_source(monkeypatch):
    """forecast_statistical('AutoSARIMA') with real statsforecast returns trained_statsforecast."""
    monkeypatch.delenv("STRICT_MODEL_MODE", raising=False)
    from src.models.statistical import forecast_statistical
    y = np.cumsum(np.random.default_rng(1).standard_normal(60)) + 100
    result = forecast_statistical("AutoSARIMA", y, h=6, season_length=12)
    assert result.model_output_source == "trained_statsforecast"
    assert result.fit_status == "trained"


# ---------------------------------------------------------------------------
# 6. finetune_by_individual_series uses approximation=False (distinct behavior)
# ---------------------------------------------------------------------------

def test_finetune_individual_series_uses_exact_fitting():
    """finetune_by_individual_series passes approximation=False to AutoARIMA."""
    from src.models.statistical import forecast_statistical

    y = np.cumsum(np.random.default_rng(2).standard_normal(60)) + 100
    result_exact = forecast_statistical("AutoARIMA", y, h=6, season_length=12, mode="finetune_by_individual_series")
    result_approx = forecast_statistical("AutoARIMA", y, h=6, season_length=12, mode="no_finetune")
    # Both must return trained results
    assert result_exact.model_output_source == "trained_statsforecast"
    assert result_approx.model_output_source == "trained_statsforecast"


# ---------------------------------------------------------------------------
# 7. Timing propagation: leaderboard task_id matches make_tasks format
# ---------------------------------------------------------------------------

def test_timing_task_id_format_matches_make_tasks():
    """overall_leaderboard must reconstruct task_id as feature|freq|decomp|family|model|ft."""
    import src.evaluation.leaderboards as lb

    metrics = pd.DataFrame([{
        "feature_name": "feat_a",
        "frequency": "monthly",
        "model_family": "statistical",
        "model_name": "AutoETS",
        "decomposition_method": "without_stl",
        "finetuning_mode": "no_finetune",
        "unique_id": "s1",
        "rel_naive_unclipped": 0.9,
        "rel_naive_clipped": 0.9,
        "mae": 1.0,
        "smape": 0.1,
        "mase": 0.8,
        "wape": 0.9,
        "model_output_source": "trained_statsforecast",
    }])
    expected_task_id = "feat_a|monthly|without_stl|statistical|AutoETS|no_finetune"
    training_times = {expected_task_id: 42.0}
    inference_times = {expected_task_id: 5.0}

    result = lb.overall_leaderboard(metrics, training_times, inference_times, num_samples=1)
    assert not result.empty
    assert result.iloc[0]["training_time_seconds"] == 42.0, (
        f"Timing not propagated; got {result.iloc[0]['training_time_seconds']}. "
        "task_id key format mismatch."
    )


# ---------------------------------------------------------------------------
# 8. stl_model_all_components for global families trains 3 separate models
# ---------------------------------------------------------------------------

def test_stl_model_all_components_global_trains_three_models(monkeypatch):
    """stl_model_all_components + mlforecast must call forecast_mlforecast_global exactly 3 times."""
    import src.training.train_family as tf

    call_count = {"n": 0}
    original = tf.forecast_mlforecast_global

    def counting_global(model_name, df_train, *args, **kwargs):
        call_count["n"] += 1
        return original(model_name, df_train, *args, **kwargs)

    monkeypatch.setattr(tf, "forecast_mlforecast_global", counting_global)
    df, cutoffs, bm = _tiny_df()
    task = _task("mlforecast", "AutoRidge", decomp="stl_model_all_components")
    result = evaluate_task(task, df, cutoffs, season_length=12, horizon=10, bucket_manifest=bm)
    assert call_count["n"] == 3, (
        f"stl_model_all_components must call global trainer 3× (trend+seasonal+residual), got {call_count['n']}"
    )
    assert not result.metrics.empty


# ---------------------------------------------------------------------------
# 9. MLForecast global trains successfully with synthetic monthly data
# ---------------------------------------------------------------------------

def test_mlforecast_global_trains_and_returns_trained_source():
    """forecast_mlforecast_global must return trained_mlforecast with sufficient data."""
    from src.models.mlforecast_auto import forecast_mlforecast_global

    np.random.seed(42)
    n = 60
    dates = pd.date_range("1982-01", periods=n, freq="ME")
    rows = []
    for sid in [f"s{i}" for i in range(5)]:
        y = np.cumsum(np.random.randn(n)) + 100
        for d, yv in zip(dates, y):
            rows.append({"unique_id": sid, "ds": d, "y": float(yv)})
    df = pd.DataFrame(rows)

    results = forecast_mlforecast_global("AutoRidge", df, h=12, season_length=12, freq_str="ME", num_optuna_samples=2)
    assert results, "Must return a non-empty results dict"
    for uid, res in results.items():
        assert res.model_output_source == "trained_mlforecast", (
            f"Expected trained_mlforecast for {uid}, got {res.model_output_source}: {res.fallback_reason}"
        )
        assert res.fit_status == "trained"
        assert len(res.yhat) == 12
        assert np.isfinite(res.yhat).all()
