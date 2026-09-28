from __future__ import annotations

FEATURE_NAMES = [
    "feature_non_normality",
    "feature_nonlinearity",
    "feature_spectral_entropy",
    "feature_evolving_seasonality",
    "feature_structural_break_strength",
    "feature_arch_stat",
]

FEATURE_ZSCORE_NAMES = [f"{name}_zscore" for name in FEATURE_NAMES]
FEATURE_BIN_NAMES = [f"{name}_bin" for name in FEATURE_NAMES]

MODEL_FAMILIES = {
    "statistical": ["AutoARIMA", "AutoSARIMA", "AutoETS"],
    "mlforecast": ["AutoRandomForest", "AutoLinearRegression", "AutoRidge", "AutoXGBoost"],
    "neuralforecast": ["AutoNLinear", "AutoNHITS", "AutoLSTM"],
    "transformers": ["AutoTFT", "AutoPatchTST"],
}

SMOKE_MODELS = {
    "statistical": ["AutoETS"],
    "mlforecast": ["AutoRidge"],
    "neuralforecast": ["AutoNLinear"],
    "transformers": ["AutoPatchTST"],
}

