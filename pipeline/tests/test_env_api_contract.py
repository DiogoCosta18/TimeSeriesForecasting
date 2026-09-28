"""Test I5 (rerun protocol, Sections 4.2-4.3): API contract of the pinned environment.

Every class, function and constructor argument the rerun relies on -- the model
classes of protocol Table 5, the tuning machinery of Section 4.3 and the
search-space parameters of Table 6 -- must exist in the pinned library versions.
A missing name here means the protocol cannot be implemented as written, so the
test must pass before any pipeline code is written against these APIs.

Arguments are checked on explicit signatures only: a name that would merely be
swallowed by **kwargs does not count as supported.
"""
from __future__ import annotations

import importlib
import inspect

import pytest


def _params(obj) -> set[str]:
    return set(inspect.signature(obj).parameters)


def _attr(module: str, name: str):
    mod = importlib.import_module(module)
    assert hasattr(mod, name), f"{module}.{name} does not exist in the pinned environment"
    return getattr(mod, name)


# --- statistical family (protocol Table 5, D11) -------------------------------------

@pytest.mark.parametrize(
    "cls, required",
    [
        ("AutoARIMA", {"season_length", "D", "approximation"}),
        ("AutoETS", {"season_length"}),
    ],
)
def test_statsforecast_models(cls, required):
    missing = required - _params(_attr("statsforecast.models", cls))
    assert not missing, f"statsforecast.models.{cls} lacks {missing}"


def test_statsforecast_engine():
    missing = {"models", "freq", "n_jobs"} - _params(_attr("statsforecast", "StatsForecast"))
    assert not missing, f"StatsForecast lacks {missing}"


# --- machine-learning family (Table 5, D25, Table 6) ---------------------------------

def test_mlforecast_engine_and_transforms():
    missing = {"models", "freq", "lags", "target_transforms"} - _params(_attr("mlforecast", "MLForecast"))
    assert not missing, f"MLForecast lacks {missing}"
    for name in ("Differences", "LocalStandardScaler"):
        _attr("mlforecast.target_transforms", name)


def test_mlforecast_auto_tuning():
    auto = _attr("mlforecast.auto", "AutoMLForecast")
    missing_init = {"models", "freq", "season_length", "init_config", "fit_config"} - _params(auto)
    assert not missing_init, f"AutoMLForecast.__init__ lacks {missing_init}"
    missing_fit = {"df", "n_windows", "h", "num_samples", "loss"} - _params(auto.fit)
    assert not missing_fit, f"AutoMLForecast.fit lacks {missing_fit}"
    missing_model = {"model", "config"} - _params(_attr("mlforecast.auto", "AutoModel"))
    assert not missing_model, f"AutoModel lacks {missing_model}"


@pytest.mark.parametrize("cls", ["AutoRidge", "AutoLinearRegression", "AutoRandomForest", "AutoXGBoost"])
def test_mlforecast_auto_models_exist(cls):
    _attr("mlforecast.auto", cls)


# --- neural and transformer families (Table 5, Table 6) ------------------------------

COMMON_NEURAL = {
    "h", "input_size", "max_steps", "learning_rate", "batch_size", "scaler_type",
    "random_seed", "valid_loss", "early_stop_patience_steps", "val_check_steps",
}

MODEL_SPECIFIC = {
    "NLinear": set(),
    "NHITS": {"n_pool_kernel_size", "n_freq_downsample", "mlp_units"},
    "LSTM": {"encoder_hidden_size", "encoder_n_layers", "decoder_hidden_size"},
    "TFT": {"hidden_size", "n_head", "dropout"},
    "PatchTST": {"hidden_size", "n_heads", "encoder_layers", "patch_len", "stride"},
}


@pytest.mark.parametrize("cls", sorted(MODEL_SPECIFIC))
def test_neuralforecast_model_arguments(cls):
    missing = (COMMON_NEURAL | MODEL_SPECIFIC[cls]) - _params(_attr("neuralforecast.models", cls))
    assert not missing, f"neuralforecast.models.{cls} lacks {missing}"


@pytest.mark.parametrize("cls", sorted(MODEL_SPECIFIC))
def test_models_support_start_padding(cls):
    assert "start_padding_enabled" in _params(_attr("neuralforecast.models", cls)), (
        f"{cls} lacks start_padding_enabled (protocol Table 6 fixes it to True)"
    )


def test_neuralforecast_engine():
    nf = _attr("neuralforecast", "NeuralForecast")
    missing = {"models", "freq"} - _params(nf)
    assert not missing, f"NeuralForecast lacks {missing}"
    assert "val_size" in _params(nf.fit), "NeuralForecast.fit lacks val_size"


@pytest.mark.parametrize("cls", ["AutoNLinear", "AutoNHITS", "AutoLSTM", "AutoTFT", "AutoPatchTST"])
def test_neuralforecast_auto_classes(cls):
    auto = _attr("neuralforecast.auto", cls)
    missing = {"h", "loss", "valid_loss", "config", "search_alg", "num_samples", "backend"} - _params(auto)
    assert not missing, f"neuralforecast.auto.{cls} lacks {missing}"
    assert "backend" in _params(auto.get_default_config), f"{cls}.get_default_config lacks backend"


def test_neuralforecast_mase_loss():
    assert "seasonality" in _params(_attr("neuralforecast.losses.pytorch", "MASE"))


# --- tuning engine (Section 4.3) ------------------------------------------------------

def test_optuna_tpe_and_sqlite_storage():
    assert "seed" in _params(_attr("optuna.samplers", "TPESampler"))
    missing = {"storage", "sampler", "direction", "study_name"} - _params(_attr("optuna", "create_study"))
    assert not missing, f"optuna.create_study lacks {missing}"


# --- decomposition (D9) -----------------------------------------------------------------

def test_statsmodels_robust_stl():
    # STL is compiled (Cython) and exposes no Python signature, so the contract is
    # checked by use: a robust fit with an explicit period must run and decompose
    # the series additively.
    import numpy as np

    stl_cls = _attr("statsmodels.tsa.seasonal", "STL")
    t = np.arange(48, dtype=float)
    y = 10 + 0.1 * t + np.sin(2 * np.pi * t / 12)
    res = stl_cls(y, period=12, robust=True).fit()
    np.testing.assert_allclose(res.trend + res.seasonal + res.resid, y, atol=1e-9)
