from __future__ import annotations

import numpy as np

from .errors import FeatureUndefined
from .mstl_features import decompose_series, finite_array, robust_scale


def structural_break_feature(y, season_length: int) -> float:
    y = finite_array(y)
    if len(y) < max(20, 4 * season_length):
        raise FeatureUndefined("too_short")
    if np.std(y) < 1e-10:
        raise FeatureUndefined("constant")
    dec = decompose_series(y, season_length)
    x = dec["trend"] + dec["residual"]
    scale = robust_scale(x)
    import ruptures as rpt

    algo = rpt.Pelt(model="rbf").fit(x.reshape(-1, 1))
    breaks = algo.predict(pen=max(3, np.log(len(x))))
    candidates = [b for b in breaks if int(0.2 * len(x)) <= b <= int(0.8 * len(x))]
    best = 0.0
    for b in candidates:
        if b <= 1 or b >= len(x) - 1:
            continue
        best = max(best, abs(np.mean(x[:b]) - np.mean(x[b:])) / scale)
    return float(np.clip(best, 0, 20))
