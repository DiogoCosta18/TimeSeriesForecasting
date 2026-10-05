from __future__ import annotations

FEATURE_NAMES = [
    "feature_non_normality",
    "feature_nonlinearity",
    "feature_spectral_entropy",
    "feature_evolving_seasonality",
    "feature_structural_break_strength",
    "feature_arch_stat",
]



def selected_features(config: dict) -> list[str]:
    """The features a run samples and evaluates, in canonical order.

    ``config["features"]`` restricts them (the pilot, Section 8.1); the default is all six.
    Eligibility always requires all six (D4), so a restricted run has the same pool.
    """
    chosen = config.get("features")
    if chosen is None:
        return list(FEATURE_NAMES)
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(FEATURE_NAMES):
        raise ValueError(f"features must be distinct names from {FEATURE_NAMES}: {chosen}")
    return [name for name in FEATURE_NAMES if name in chosen]

