"""Exceptions of the feature and decomposition layer (protocol D4, D9; defects F1, F2)."""
from __future__ import annotations


class FeatureUndefined(ValueError):
    """A feature is not defined for this series (too short, constant, degenerate).

    Raised instead of returning a substitute value: the series gets a non-"ok"
    quality flag naming ``reason`` and is ineligible (protocol D4, defect F1).
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class DecompositionError(FeatureUndefined):
    """Robust STL cannot be fitted to this input. There is no fallback decomposition
    (protocol D9, defect F2): a feature that needs STL is undefined, and a forecast
    that needs STL fails."""
