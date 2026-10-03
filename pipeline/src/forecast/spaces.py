"""Search spaces of protocol Table 6 (D14, D25), written once for tuning and evaluation.

``sample_*`` draws one point of a space from an Optuna trial and returns plain, JSON-safe
parameters (these are what a frozen configuration stores). ``*_kwargs`` / ``ml_*`` turn
parameters into the arguments of the library classes. Tuning and evaluation both go
through these functions, so a frozen configuration builds exactly the model that was
validated. Nothing here depends on the data or on the composition of a training pool
(defect C4): the arguments are a function of the parameters, h, m and the seed only.
"""
from __future__ import annotations

from src.forecast.registry import family

INPUT_SIZE = {12: [18, 36], 4: [8, 16]}           # monthly / quarterly
MAX_STEPS = [500, 1000, 2000]
BATCH_SIZE = [32, 64]
SCALER = ["identity", "standard", "robust"]
TARGET_TRANSFORMS = ["none", "local_standard", "first_difference", "seasonal_difference"]
XGBOOST_THREADS = 4  # fixed, so that results do not depend on the machine


# --- neural and transformer families ------------------------------------------------------

def sample_neural(model: str, trial, season_length: int) -> dict:
    p = {
        "input_size": trial.suggest_categorical("input_size", INPUT_SIZE[season_length]),
        "scaler_type": trial.suggest_categorical("scaler_type", SCALER),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "max_steps": trial.suggest_categorical("max_steps", MAX_STEPS),
        "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZE),
    }
    if model == "NHITS":
        p["n_pool_kernel_size"] = trial.suggest_categorical("n_pool_kernel_size", ["2,2,1", "1,1,1"])
        p["n_freq_downsample"] = trial.suggest_categorical("n_freq_downsample", ["4,2,1", "1,1,1"])
        p["mlp_units"] = trial.suggest_categorical("mlp_units", ["2x64", "2x256"])
    elif model == "LSTM":
        p["encoder_hidden_size"] = trial.suggest_categorical("encoder_hidden_size", [32, 64, 128])
        p["encoder_n_layers"] = trial.suggest_categorical("encoder_n_layers", [1, 2])
        p["decoder_hidden_size"] = trial.suggest_categorical("decoder_hidden_size", [32, 64])
    elif model == "TFT":
        p["hidden_size"] = trial.suggest_categorical("hidden_size", [32, 64, 128])
        p["n_head"] = trial.suggest_categorical("n_head", [2, 4])
        p["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
    elif model == "PatchTST":
        p["hidden_size"] = trial.suggest_categorical("hidden_size", [32, 64, 128])
        p["n_heads"] = trial.suggest_categorical("n_heads", [4, 8])
        p["encoder_layers"] = trial.suggest_categorical("encoder_layers", [1, 2, 3])
        p["patch_len"] = trial.suggest_categorical("patch_len", [4, 8])
    elif model != "NLinear":
        raise KeyError(f"{model} is not a neural or transformer model")
    return p


def neural_kwargs(model: str, params: dict, h: int, seed: int, *, max_steps: int, accelerator: str,
                  early_stopping: dict | None = None) -> dict:
    """Arguments of the plain neuralforecast class for ``params``.

    ``max_steps`` is the searched value during tuning and the frozen number of trained
    steps in evaluation; ``early_stopping`` ({patience, check_steps}) is given only when
    tuning (protocol Section 4.3).
    """
    kw = {
        "h": h,
        "input_size": int(params["input_size"]),
        "scaler_type": params["scaler_type"],
        "learning_rate": float(params["learning_rate"]),
        "max_steps": int(max_steps),
        "batch_size": int(params["batch_size"]),
        "start_padding_enabled": True,  # Table 6: fixed
        "random_seed": int(seed),
        "enable_progress_bar": False,
        "logger": False,
        "accelerator": accelerator,
        "devices": 1,
        # Bit-for-bit reproducible on CPU (test U10). On a GPU some operations (e.g. the median
        # of the robust scaler) have no deterministic CUDA kernel: deterministic where possible,
        # warn otherwise; the seed check (A15) quantifies the remaining variation.
        "deterministic": True if accelerator == "cpu" else "warn",
    }
    if early_stopping:
        kw["early_stop_patience_steps"] = int(early_stopping["patience"])
        kw["val_check_steps"] = int(early_stopping["check_steps"])
    if model == "NHITS":
        kw["n_pool_kernel_size"] = [int(v) for v in params["n_pool_kernel_size"].split(",")]
        kw["n_freq_downsample"] = [int(v) for v in params["n_freq_downsample"].split(",")]
        units = int(params["mlp_units"].split("x")[1])
        kw["mlp_units"] = [[units, units]] * 3
    elif model == "LSTM":
        kw.update({k: int(params[k]) for k in ("encoder_hidden_size", "encoder_n_layers", "decoder_hidden_size")})
    elif model == "TFT":
        kw.update(hidden_size=int(params["hidden_size"]), n_head=int(params["n_head"]), dropout=float(params["dropout"]))
    elif model == "PatchTST":
        patch_len = min(int(params["patch_len"]), kw["input_size"])  # Table 6: capped at input_size
        kw.update(hidden_size=int(params["hidden_size"]), n_heads=int(params["n_heads"]),
                  encoder_layers=int(params["encoder_layers"]), patch_len=patch_len, stride=patch_len // 2)
    return kw


def neural_class(model: str):
    from neuralforecast import models as nf_models

    return getattr(nf_models, model)


def neural_auto_class(model: str):
    from neuralforecast import auto as nf_auto

    return getattr(nf_auto, f"Auto{model}")


# --- machine-learning family (D25) ----------------------------------------------------------

def sample_ml(model: str, trial) -> dict:
    p = {"target_transform": trial.suggest_categorical("target_transform", TARGET_TRANSFORMS)}
    if model == "Ridge":
        p["alpha"] = trial.suggest_float("alpha", 1e-3, 1e2, log=True)
    elif model == "RandomForest":
        p["n_estimators"] = trial.suggest_categorical("n_estimators", [100, 300])
        p["max_depth"] = trial.suggest_categorical("max_depth", ["none", 10, 20])
        p["min_samples_leaf"] = trial.suggest_categorical("min_samples_leaf", [1, 5, 20])
        p["max_features"] = trial.suggest_categorical("max_features", [1.0, 0.5, "sqrt"])
    elif model == "XGBoost":
        p["n_estimators"] = trial.suggest_int("n_estimators", 100, 600)
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        p["max_depth"] = trial.suggest_int("max_depth", 3, 9)
        p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    elif model != "LinearRegression":
        raise KeyError(f"{model} is not a machine-learning model")
    return p


def ml_estimator(model: str, params: dict, seed: int):
    """The scikit-learn compatible estimator for ``params`` (target transform excluded)."""
    if family(model) != "ml":
        raise KeyError(f"{model} is not a machine-learning model")
    if model == "Ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=float(params["alpha"]))
    if model == "LinearRegression":
        from sklearn.linear_model import LinearRegression

        return LinearRegression()
    if model == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor

        depth = None if params["max_depth"] == "none" else int(params["max_depth"])
        return RandomForestRegressor(
            n_estimators=int(params["n_estimators"]), max_depth=depth,
            min_samples_leaf=int(params["min_samples_leaf"]), max_features=params["max_features"],
            # One thread: with several, the tree predictions are summed in thread-completion
            # order and the last bits of the forecasts vary between identical runs.
            random_state=int(seed), n_jobs=1,
        )
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=int(params["n_estimators"]), learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]), subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]), random_state=int(seed),
        n_jobs=XGBOOST_THREADS, tree_method="hist", verbosity=0,
    )


def ml_lags(season_length: int) -> list[int]:
    """D25: lags 1..m, fixed."""
    return list(range(1, int(season_length) + 1))


def ml_target_transforms(params: dict, season_length: int) -> list:
    from mlforecast.target_transforms import Differences, LocalStandardScaler

    choice = params["target_transform"]
    if choice == "none":
        return []
    if choice == "local_standard":
        return [LocalStandardScaler()]
    if choice == "first_difference":
        return [Differences([1])]
    if choice == "seasonal_difference":
        return [Differences([int(season_length)])]
    raise KeyError(f"unknown target transform {choice!r}")
