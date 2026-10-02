from __future__ import annotations

import numpy as np

from .errors import FeatureUndefined
from .mstl_features import decompose_series, finite_array, robust_scale


def arch_feature(y, season_length: int) -> float:
    y = finite_array(y)
    # Guard: monthly 48 (4 x 12); quarterly 20, lowered from 24 by protocol D5.
    if len(y) < max(20, 4 * season_length):
        raise FeatureUndefined("too_short")
    if np.std(y) < 1e-10:
        raise FeatureUndefined("constant")
    resid = decompose_series(y, season_length)["residual"]
    z = (resid - np.median(resid)) / robust_scale(resid)
    from statsmodels.stats.diagnostic import het_arch

    _, p, *_ = het_arch(z, nlags=min(12, max(1, len(z) // 5)))
    return float(np.clip(-np.log10(float(p) + 1e-12), 0, 12))
