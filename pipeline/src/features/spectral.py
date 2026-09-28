from __future__ import annotations

import numpy as np

from .mstl_features import clean_array


def spectral_entropy_feature(y) -> tuple[float, str]:
    y = clean_array(y)
    if len(y) < 8 or np.std(y) < 1e-10:
        return 1.0, "fallback_constant_or_invalid"
    try:
        from scipy.signal import periodogram

        _, pxx = periodogram(y - np.mean(y))
        pxx = pxx[pxx > 0]
        if len(pxx) <= 1:
            return 1.0, "fallback_constant_or_invalid"
        probs = pxx / pxx.sum()
        h = -np.sum(probs * np.log(probs)) / np.log(len(probs))
        return float(np.clip(h, 0, 1)), "ok"
    except Exception:
        return 1.0, "fallback_constant_or_invalid"

