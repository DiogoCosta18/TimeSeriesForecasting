"""The 12 models (protocol D10, Table 5), the strategies and the targets they forecast."""
from __future__ import annotations

FAMILY = {
    "ARIMA": "statistical", "SARIMA": "statistical", "ETS": "statistical",
    "Ridge": "ml", "LinearRegression": "ml", "RandomForest": "ml", "XGBoost": "ml",
    "NLinear": "neural", "NHITS": "neural", "LSTM": "neural",
    "TFT": "transformer", "PatchTST": "transformer",
}
MODELS = list(FAMILY)
FAMILIES = ["statistical", "ml", "neural", "transformer"]
GLOBAL_MODELS = [name for name, fam in FAMILY.items() if fam != "statistical"]
STATISTICAL_MODELS = [name for name, fam in FAMILY.items() if fam == "statistical"]
LINEARITY = {  # Table 5, last column
    "Ridge": "linear", "LinearRegression": "linear", "RandomForest": "nonlinear", "XGBoost": "nonlinear",
    "NLinear": "linear", "NHITS": "nonlinear", "LSTM": "nonlinear", "TFT": "nonlinear", "PatchTST": "nonlinear",
}

STRATEGIES = ["direct", "stl_sn", "stl_ac"]
# What the model forecasts under each strategy (protocol Section 4.1): the raw series;
# T + R with the last seasonal cycle added back; or T, S and R separately, summed.
STRATEGY_TARGETS = {"direct": ["raw"], "stl_sn": ["nonseasonal"], "stl_ac": ["trend", "seasonal", "residual"]}
TARGETS = ["raw", "nonseasonal", "trend", "seasonal", "residual"]

SOURCE = {"statistical": "trained_statsforecast", "ml": "trained_mlforecast",
          "neural": "trained_neuralforecast", "transformer": "trained_neuralforecast"}
BACKEND = {"statistical": "statsforecast", "ml": "mlforecast", "neural": "neuralforecast", "transformer": "neuralforecast"}


def family(model: str) -> str:
    if model not in FAMILY:
        raise KeyError(f"unknown model {model!r}; the protocol's models are {MODELS}")
    return FAMILY[model]


def config_key(model: str, frequency: str, target: str) -> str:
    """Key of a frozen configuration: one study per global model x frequency x target (D13)."""
    if model not in GLOBAL_MODELS or target not in TARGETS or frequency not in ("monthly", "quarterly"):
        raise KeyError(f"no frozen configuration for {model}/{frequency}/{target}")
    return f"{model}|{frequency}|{target}"
