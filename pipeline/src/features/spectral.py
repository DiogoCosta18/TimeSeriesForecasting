from __future__ import annotations

import numpy as np

from .errors import FeatureUndefined
from .mstl_features import finite_array


def spectral_entropy_feature(y) -> float:
    y = finite_array(y)
    if len(y) < 8:
        raise FeatureUndefined("too_short")
    if np.std(y) < 1e-10:
        raise FeatureUndefined("constant")
    from scipy.signal import periodogram

    _, pxx = periodogram(y - np.mean(y))
    pxx = pxx[pxx > 0]
    if len(pxx) <= 1:
        raise FeatureUndefined("degenerate_spectrum")
    probs = pxx / pxx.sum()
    h = -np.sum(probs * np.log(probs)) / np.log(len(probs))
    return float(np.clip(h, 0, 1))
