"""Model layer: result object, search spaces, frozen-configuration fits (tests U9, U10, U11)."""
from __future__ import annotations

import ast
import inspect

import numpy as np
import optuna
import pandas as pd
import pytest

from src.forecast import metrics, models, registry, result, spaces, targets, tuning
from src.forecast.registry import FAMILY, GLOBAL_MODELS, MODELS, STATISTICAL_MODELS
from src.forecast.result import TRAINED_SOURCES, ForecastError, ForecastResult

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 20260521


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    """U10 is defined on CPU backends (neural with a fixed seed on CPU)."""
    monkeypatch.setenv("RERUN_ACCELERATOR", "cpu")


def pool(n_series: int, lengths: tuple[int, int], m: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_series):
        n = int(rng.integers(*lengths))
        t = np.arange(1, n + 1)
        y = 100 + 0.4 * t + 6 * np.sin(2 * np.pi * t / m + i) + rng.normal(0, 1.5, n)
        rows += [{"unique_id": f"S{i:03d}", "ds": int(tt), "y": float(v)} for tt, v in zip(t, y)]
    return pd.DataFrame(rows)


def sampled_params(model: str, m: int, seed: int) -> dict:
    trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed)).ask()
    return spaces.sample_ml(model, trial) if FAMILY[model] == "ml" else spaces.sample_neural(model, trial, m)


# --- U9: only trained forecasts exist ----------------------------------------------------------

@pytest.mark.parametrize("source", ["deterministic_baseline_fallback", "placeholder_optuna", "failed_fallback",
                                    "strict_mode_failure", "skipped", "seasonal_naive"])
def test_u9_result_object_rejects_any_non_trained_source(source):
    with pytest.raises(ForecastError, match="not a trained model"):
        ForecastResult(np.ones(4), source, "x", "1", 4)


def test_u9_result_object_rejects_wrong_length_or_non_finite_values():
    ForecastResult(np.ones(4), "trained_mlforecast", "mlforecast", "1", 4)
    with pytest.raises(ForecastError, match="expected 4"):
        ForecastResult(np.ones(3), "trained_mlforecast", "mlforecast", "1", 4)
    with pytest.raises(ForecastError, match="non-finite"):
        ForecastResult(np.array([1.0, np.nan, 1.0, 1.0]), "trained_mlforecast", "mlforecast", "1", 4)
    assert TRAINED_SOURCES == {registry.SOURCE[f] for f in registry.FAMILIES}


@pytest.mark.parametrize("module", [models, spaces, result, registry, tuning, targets, metrics])
def test_u9_model_layer_has_no_recovery_or_stub_paths(module):
    source = inspect.getsource(module)
    tree = ast.parse(source)
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not {m for m in imported if m.startswith(("src.models", "src.training"))}
    for node in ast.walk(tree):  # drop docstrings: words may be named there to say they are absent
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
            body.pop(0)
    code = ast.unparse(tree)  # code only: comments are gone too
    for word in ("fallback", "baseline", "seasonal_naive", "identity_scale", "jitter", "_safe_"):
        assert word not in code


def test_registry_holds_the_twelve_models_of_the_protocol():
    assert MODELS == ["ARIMA", "SARIMA", "ETS", "Ridge", "LinearRegression", "RandomForest", "XGBoost",
                      "NLinear", "NHITS", "LSTM", "TFT", "PatchTST"]
    assert len(GLOBAL_MODELS) == 9 and STATISTICAL_MODELS == ["ARIMA", "SARIMA", "ETS"]
    assert registry.config_key("NHITS", "monthly", "trend") == "NHITS|monthly|trend"
    with pytest.raises(KeyError):
        registry.config_key("ARIMA", "monthly", "raw")  # statistical models are not tuned


# --- search spaces build valid models -----------------------------------------------------------

@pytest.mark.parametrize("model", GLOBAL_MODELS)
@pytest.mark.parametrize("m", [12, 4])
def test_every_sampled_configuration_builds_a_valid_model(model, m):
    for seed in range(15):
        params = sampled_params(model, m, seed)
        if FAMILY[model] == "ml":
            est = spaces.ml_estimator(model, params, SEED)
            if model in ("RandomForest", "XGBoost"):
                assert est.get_params()["random_state"] == SEED
            spaces.ml_target_transforms(params, m)
        else:
            kw = spaces.neural_kwargs(model, params, h=8, seed=SEED, max_steps=params["max_steps"], accelerator="cpu")
            spaces.neural_class(model)(**kw)  # constructs without error
            assert kw["input_size"] in spaces.INPUT_SIZE[m] and kw["start_padding_enabled"] is True


def test_spaces_follow_table_6():
    assert spaces.INPUT_SIZE == {12: [18, 36], 4: [8, 16]}
    assert spaces.MAX_STEPS == [500, 1000, 2000] and spaces.BATCH_SIZE == [32, 64]
    assert spaces.SCALER == ["identity", "standard", "robust"]
    assert spaces.ml_lags(12) == list(range(1, 13)) and spaces.ml_lags(4) == [1, 2, 3, 4]
    kw = spaces.neural_kwargs("PatchTST", {"input_size": 8, "scaler_type": "robust", "learning_rate": 1e-3,
                                           "max_steps": 500, "batch_size": 32, "hidden_size": 64, "n_heads": 4,
                                           "encoder_layers": 2, "patch_len": 8}, 8, SEED, max_steps=500, accelerator="cpu")
    assert (kw["patch_len"], kw["stride"]) == (8, 4)
    kw = spaces.neural_kwargs("NHITS", {"input_size": 36, "scaler_type": "standard", "learning_rate": 1e-3,
                                        "max_steps": 500, "batch_size": 32, "n_pool_kernel_size": "2,2,1",
                                        "n_freq_downsample": "4,2,1", "mlp_units": "2x256"}, 18, SEED,
                              max_steps=500, accelerator="cpu")
    assert kw["mlp_units"] == [[256, 256]] * 3 and kw["n_pool_kernel_size"] == [2, 2, 1]


# --- U10: same task, same forecasts (CPU) ---------------------------------------------------------

@pytest.mark.parametrize("model", GLOBAL_MODELS)
def test_u10_the_same_fit_twice_gives_identical_forecast_hashes(model):
    m, h = 4, 8
    train = pool(12, (30, 50), m, seed=1)
    params = sampled_params(model, m, seed=3)
    steps = 4 if FAMILY[model] != "ml" else None
    a = models.fit_predict_global(model, params, train, h, m, SEED, trained_steps=steps)
    b = models.fit_predict_global(model, params, train, h, m, SEED, trained_steps=steps)
    assert sorted(a) == sorted(train["unique_id"].unique())
    assert [a[k].forecast_hash for k in sorted(a)] == [b[k].forecast_hash for k in sorted(b)]
    assert all(r.source == registry.SOURCE[FAMILY[model]] and r.seed == SEED for r in a.values())


@pytest.mark.parametrize("model", STATISTICAL_MODELS)
def test_u10_statistical_forecasts_are_reproducible(model):
    y = pool(1, (60, 61), 12, seed=2)["y"].to_numpy()
    a = models.forecast_statistical(model, y, 18, 12)
    b = models.forecast_statistical(model, y, 18, 12)
    assert a.forecast_hash == b.forecast_hash and a.source == "trained_statsforecast" and len(a.yhat) == 18


@pytest.mark.parametrize("model", STATISTICAL_MODELS)
@pytest.mark.parametrize("m, h", [(12, 18), (4, 8)])
def test_statistical_fit_predict_equals_the_earlier_runs_forecast_call(model, m, h):
    """Regression test I2 needs the May run's numbers: fit + predict must equal forecast() exactly."""
    from statsforecast import StatsForecast

    data = pool(8, (3 * m + h, 3 * m + h + 60), m, seed=m)
    for _, g in data.groupby("unique_id"):
        y = g["y"].to_numpy()
        df = pd.DataFrame({"unique_id": ["s"] * len(y), "ds": range(len(y)), "y": y})
        reference = StatsForecast(models=[models.statistical_model(model, m)], freq=1, n_jobs=1).forecast(df=df, h=h)
        ours = models.forecast_statistical(model, y, h, m)
        np.testing.assert_array_equal(ours.yhat, reference.iloc[:, -1].to_numpy(dtype=float))


def test_statistical_failure_raises_instead_of_substituting():
    with pytest.raises(Exception):
        models.forecast_statistical("ARIMA", np.array([]), 8, 4)


# --- U11: the frozen configuration, whatever the pool ---------------------------------------------

def test_u11_ml_models_get_exactly_the_frozen_configuration_whatever_the_pool(monkeypatch):
    seen = []
    original = spaces.ml_estimator

    def recording(model, params, seed):
        est = original(model, params, seed)
        seen.append(est.get_params())
        return est

    monkeypatch.setattr(spaces, "ml_estimator", recording)
    params = sampled_params("XGBoost", 4, seed=4)
    short_pool, long_pool = pool(8, (20, 22), 4, seed=5), pool(40, (60, 120), 4, seed=6)
    models.fit_predict_global("XGBoost", params, short_pool, 8, 4, SEED)
    models.fit_predict_global("XGBoost", params, long_pool, 8, 4, SEED)
    assert seen[0] == seen[1]


def test_u11_neural_models_get_exactly_the_frozen_configuration_whatever_the_pool(monkeypatch):
    seen = []
    original = spaces.neural_class

    def recording_class(model):
        cls = original(model)

        def build(**kwargs):
            seen.append(dict(kwargs))
            return cls(**kwargs)

        return build

    monkeypatch.setattr(spaces, "neural_class", recording_class)
    params = sampled_params("NHITS", 4, seed=7)
    models.fit_predict_global("NHITS", params, pool(6, (20, 22), 4, seed=8), 8, 4, SEED, trained_steps=3)
    models.fit_predict_global("NHITS", params, pool(30, (60, 120), 4, seed=9), 8, 4, SEED, trained_steps=3)
    assert len(seen) == 2 and seen[0] == seen[1]
    assert seen[0]["input_size"] == params["input_size"] and seen[0]["max_steps"] == 3


def test_neural_fit_requires_the_frozen_number_of_steps():
    with pytest.raises(ForecastError, match="training steps"):
        models.fit_predict_global("NLinear", sampled_params("NLinear", 4, 1), pool(4, (30, 31), 4, 1), 8, 4, SEED)
