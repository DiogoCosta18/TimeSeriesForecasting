from __future__ import annotations

import numpy as np

from .mstl_features import clean_array, decompose_series, robust_scale


def structural_break_feature(y, season_length: int) -> tuple[float, str]:
    y = clean_array(y)
    if len(y) < max(20, 4 * season_length) or np.std(y) < 1e-10:
        return 0.0, "fallback_short"
    dec = decompose_series(y, season_length)
    x = dec["trend"] + dec["residual"]
    scale = robust_scale(x)
    try:
        import ruptures as rpt

        algo = rpt.Pelt(model="rbf").fit(x.reshape(-1, 1))
        breaks = algo.predict(pen=max(3, np.log(len(x))))
        candidates = [b for b in breaks if int(0.2 * len(x)) <= b <= int(0.8 * len(x))]
    except Exception:
        candidates = range(int(0.2 * len(x)), int(0.8 * len(x)))
    best = 0.0
    for b in candidates:
        if b <= 1 or b >= len(x) - 1:
            continue
        best = max(best, abs(np.mean(x[:b]) - np.mean(x[b:])) / scale)
    return float(np.clip(best, 0, 20)), "ok"

