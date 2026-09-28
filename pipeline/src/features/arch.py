from __future__ import annotations

import numpy as np

from .mstl_features import clean_array, decompose_series, robust_scale


def arch_feature(y, season_length: int) -> tuple[float, str]:
    y = clean_array(y)
    if len(y) < max(24, 4 * season_length) or np.std(y) < 1e-10:
        return 0.0, "fallback_short"
    resid = decompose_series(y, season_length)["residual"]
    z = (resid - np.median(resid)) / robust_scale(resid)
    try:
        from statsmodels.stats.diagnostic import het_arch

        _, p, *_ = het_arch(z, nlags=min(12, max(1, len(z) // 5)))
        return float(np.clip(-np.log10(float(p) + 1e-12), 0, 12)), "ok"
    except Exception:
        sq = z**2
        if len(sq) < 3:
            return 0.0, "fallback_short"
        ac = np.corrcoef(sq[:-1], sq[1:])[0, 1]
        return float(np.clip(abs(ac) * 6, 0, 12)), "fallback_squared_acf"

