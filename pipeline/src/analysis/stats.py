"""Statistical procedures of the analysis plan (protocol Sections 5.3-5.4; D23, D26, D27; test U19).

- Wilcoxon signed-rank on per-series differences (two-sided; zero differences dropped,
  as in Wilcoxon's original procedure); a contrast without any non-zero difference has
  p = 1 (no evidence).
- Matched-pairs rank-biserial correlation (Kerby 2014): (R+ - R-) / (R+ + R-), ranks of
  |d| over the non-zero differences, ties averaged.
- Holm's step-down adjustment within a stated family of tests.
- Percentile bootstrap confidence intervals over series (2,000 resamples), seeded per
  contrast from the run's base seed, so every interval is reproducible.
- Jonckheere-Terpstra ordered-alternative test with the tie-corrected null variance
  (Hollander, Wolfe and Chicken 2014, Section 6.2) and a two-sided normal p-value.
- Friedman test with distinct series as blocks (ranks within blocks, ties averaged,
  rank 1 = lowest error) and Nemenyi's critical difference
  CD = q_alpha * sqrt(k (k + 1) / (6 N)), q_alpha the studentized range quantile / sqrt(2).
- D26 labels: gain / loss if Holm-significant and |median per-series delta| >= 0.01,
  negligible if significant but smaller, no evidence otherwise.
"""
from __future__ import annotations

import zlib

import numpy as np
from scipy import stats

ALPHA = 0.05
PRACTICAL_THRESHOLD = 0.01   # D26
N_BOOTSTRAP = 2000
CI_LEVEL = 0.95
LABELS = ("gain", "loss", "negligible", "no evidence")


def holm(pvalues) -> np.ndarray:
    """Holm-adjusted p-values (monotone, capped at 1), in the input order."""
    p = np.asarray(pvalues, dtype=float)
    if np.isnan(p).any():
        raise ValueError("Holm adjustment of a missing p-value")
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for i, idx in enumerate(np.argsort(p, kind="mergesort")):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adjusted[idx] = running
    return adjusted


def rank_biserial(d) -> float:
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    plus, minus = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((plus - minus) / (plus + minus))


def signed_rank(d) -> dict:
    """Wilcoxon signed-rank test of per-series differences against zero."""
    d = np.asarray(d, dtype=float)
    if not np.isfinite(d).all():
        raise ValueError("non-finite per-series difference")
    nonzero = d[d != 0]
    if len(nonzero) == 0:
        return {"n_series": len(d), "n_nonzero": 0, "statistic": np.nan, "p": 1.0, "rank_biserial": 0.0}
    res = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", method="auto")
    return {"n_series": len(d), "n_nonzero": len(nonzero), "statistic": float(res.statistic), "p": float(res.pvalue),
            "rank_biserial": rank_biserial(nonzero)}


def bootstrap_rng(base_seed: int, contrast_id: str) -> np.random.Generator:
    """One reproducible stream per contrast, derived from the base seed and the contrast's name."""
    return np.random.default_rng(np.random.SeedSequence([int(base_seed), zlib.crc32(contrast_id.encode("utf-8"))]))


def bootstrap_ci(x, statistic, rng: np.random.Generator, n_resamples: int = N_BOOTSTRAP, level: float = CI_LEVEL,
                 chunk: int = 200) -> tuple[float, float]:
    """Percentile interval of ``statistic(sample, axis=1)`` over resampled series."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    values = []
    for start in range(0, n_resamples, chunk):
        idx = rng.integers(0, len(x), size=(min(chunk, n_resamples - start), len(x)))
        values.append(statistic(x[idx], axis=1))
    values = np.concatenate(values)
    tail = (1 - level) / 2
    return float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))


def median_ci(x, base_seed: int, contrast_id: str) -> tuple[float, float]:
    return bootstrap_ci(x, np.median, bootstrap_rng(base_seed, contrast_id))


def spearman(x, y, base_seed: int, contrast_id: str, n_resamples: int = N_BOOTSTRAP) -> dict:
    """Spearman's rho with its p-value and a percentile bootstrap CI over series (pairs)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    res = stats.spearmanr(x, y)
    rng = bootstrap_rng(base_seed, contrast_id)
    rhos = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, len(x), size=len(x))
        rx, ry = stats.rankdata(x[idx]), stats.rankdata(y[idx])
        rhos[i] = np.corrcoef(rx, ry)[0, 1] if rx.std() > 0 and ry.std() > 0 else np.nan
    tail = (1 - CI_LEVEL) / 2
    rhos = rhos[np.isfinite(rhos)]
    return {"n_series": len(x), "rho": float(res.statistic), "p": float(res.pvalue),
            "rho_ci_low": float(np.quantile(rhos, tail)), "rho_ci_high": float(np.quantile(rhos, 1 - tail))}


def jonckheere_terpstra(groups) -> dict:
    """Test of an increasing trend across ordered groups (two-sided normal approximation).

    J = sum over group pairs i < j of #(x_i < x_j) + 0.5 #(x_i == x_j); its null mean is
    (N^2 - sum n_i^2) / 4 and its null variance is corrected for ties.
    """
    groups = [np.asarray(g, dtype=float) for g in groups]
    if sum(len(g) > 0 for g in groups) < 2:
        raise ValueError("the trend test needs at least two non-empty groups")
    groups = [g for g in groups if len(g) > 0]
    j = 0.0
    for i in range(len(groups)):
        lower = np.sort(groups[i])
        for later in groups[i + 1:]:
            less = np.searchsorted(lower, later, side="left")
            less_equal = np.searchsorted(lower, later, side="right")
            j += float(less.sum() + 0.5 * (less_equal - less).sum())
    n = np.array([len(g) for g in groups], dtype=float)
    N = n.sum()
    _, t = np.unique(np.concatenate(groups), return_counts=True)
    t = t.astype(float)
    mean = (N ** 2 - (n ** 2).sum()) / 4
    var = ((N * (N - 1) * (2 * N + 5) - (n * (n - 1) * (2 * n + 5)).sum() - (t * (t - 1) * (2 * t + 5)).sum()) / 72
           + (n * (n - 1) * (n - 2)).sum() * (t * (t - 1) * (t - 2)).sum() / (36 * N * (N - 1) * (N - 2))
           + (n * (n - 1)).sum() * (t * (t - 1)).sum() / (8 * N * (N - 1)))
    z = (j - mean) / np.sqrt(var) if var > 0 else 0.0
    return {"statistic": j, "mean": float(mean), "variance": float(var), "z": float(z),
            "p": float(2 * stats.norm.sf(abs(z)))}


def friedman(matrix) -> dict:
    """Friedman test on a blocks x configurations matrix of errors (complete blocks)."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 3 or m.shape[0] < 2 or not np.isfinite(m).all():
        raise ValueError("Friedman needs a complete matrix with >= 2 blocks and >= 3 configurations")
    res = stats.friedmanchisquare(*m.T)
    ranks = stats.rankdata(m, axis=1)
    return {"n_blocks": m.shape[0], "k": m.shape[1], "statistic": float(res.statistic), "p": float(res.pvalue),
            "mean_ranks": ranks.mean(axis=0)}


def nemenyi_q(k: int, alpha: float = ALPHA) -> float:
    return float(stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))


def nemenyi_cd(k: int, n_blocks: int, alpha: float = ALPHA) -> float:
    return nemenyi_q(k, alpha) * float(np.sqrt(k * (k + 1) / (6 * n_blocks)))


def d26_label(p_holm: float, effect: float) -> str:
    if not p_holm < ALPHA:
        return "no evidence"
    if abs(effect) < PRACTICAL_THRESHOLD:
        return "negligible"
    return "gain" if effect > 0 else "loss"


def weighted_median(values, weights) -> float:
    """Median with weights; equal weights give numpy's median (middle pair averaged)."""
    v, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    order = np.argsort(v, kind="mergesort")
    v, cw = v[order], np.cumsum(w[order]) / w.sum()
    i = int(np.searchsorted(cw, 0.5 - 1e-12, side="left"))
    if np.isclose(cw[i], 0.5) and i + 1 < len(v):
        return float((v[i] + v[i + 1]) / 2)
    return float(v[i])


def describe(values, weights=None) -> dict:
    """Instance-level descriptives (D27): mean, median, win rate (share > 0), tie share."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "win_rate": np.nan, "tie_share": np.nan}
    w = np.ones(len(v)) if weights is None else np.asarray(weights, dtype=float)
    return {"n": len(v), "mean": float(np.average(v, weights=w)),
            "median": float(np.median(v)) if weights is None else weighted_median(v, w),
            "win_rate": float(np.average(v > 0, weights=w)), "tie_share": float(np.average(v == 0, weights=w))}
