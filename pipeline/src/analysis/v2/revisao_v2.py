"""Reference implementations for the v2 revision analyses.

These functions fix every computational choice of the revision analyses. They are
deterministic given their inputs and seed.

Requires numpy, pandas, scipy and statsmodels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

DELTA = 0.01            # practical threshold, in units of the seasonal naive MAE
CAP = 10.0              # cap of RelNaive before averaging (frozen run)
ALPHA = 0.05            # family-wise error rate of each question family
BOOT_SEED = 20261024    # seed of every bootstrap in the revision
LOG_DELTA = -np.log(0.99)  # threshold on the log scale: 1% of the Direct MAE


# ---------------------------------------------------------------------------
# Metric and reference forecasts
# ---------------------------------------------------------------------------

def relnaive_capped(mae, mae_naive, cap=CAP):
    """RelNaive capped at `cap`, as in the frozen run (Eq. 8 and the cap)."""
    return np.minimum(np.asarray(mae, float) / np.asarray(mae_naive, float), cap)


def seasonal_naive(y_train, m, h):
    """Seasonal naive forecast y_{t+j-m*ceil(j/m)}, j = 1..h (also the S add-back)."""
    y = np.asarray(y_train, float)
    j = np.arange(1, h + 1)
    idx = len(y) - 1 + j - m * np.ceil(j / m).astype(int)
    return y[idx]


def mean_last_cycles(s_train, m, h, k):
    """Add-back diagnostic: mean of the last k cycles of S, repeated over h steps."""
    s = np.asarray(s_train, float)
    cycles = s[len(s) - k * m:].reshape(k, m)       # oldest cycle first
    pattern = cycles.mean(axis=0)                     # one value per cycle position
    j = np.arange(1, h + 1)
    return pattern[(j - 1) % m]


def insample_scale(y_train, m):
    """MASE scale: in-sample MAE of the seasonal naive (lag m) on the training part."""
    y = np.asarray(y_train, float)
    return np.mean(np.abs(y[m:] - y[:-m]))


# ---------------------------------------------------------------------------
# Seasonality test (R code of the M4 benchmarks, SeasonalityTest)
# ---------------------------------------------------------------------------

def acf(x, nlags):
    """Sample autocorrelation r_0..r_nlags with the usual biased estimator (as R acf)."""
    x = np.asarray(x, float)
    xc = x - x.mean()
    d = np.dot(xc, xc)
    if d == 0:
        return np.full(nlags + 1, np.nan)
    return np.array([1.0] + [np.dot(xc[k:], xc[:-k]) / d for k in range(1, nlags + 1)])


def seasonality_test(x, m, tcrit=1.645):
    """True if |r_m| exceeds the 90% two-sided Bartlett limit.

    limit = tcrit * sqrt((1 + 2 * sum_{k=1}^{m-1} r_k^2) / n); False if n < 3m,
    if m <= 1 or if the series is constant.
    """
    x = np.asarray(x, float)
    n = len(x)
    if m <= 1 or n < 3 * m:
        return False
    r = acf(x, m)
    if np.isnan(r).any():
        return False
    limit = tcrit * np.sqrt((1 + 2 * np.sum(r[1:m] ** 2)) / n)
    return bool(abs(r[m]) > limit)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

LABELS = ("gain", "loss", "negligible", "positive", "negative", "inconclusive")


def label(lo, hi, delta=DELTA):
    """Label of an interval [lo, hi] for an effect where positive means 'helps'.

    gain:       lo >= delta
    loss:       hi <= -delta
    negligible: -delta < lo and hi < delta
    positive:   lo > 0 (and hi >= delta)      -> improves, size uncertain
    negative:   hi < 0 (and lo <= -delta)     -> worsens, size uncertain
    inconclusive otherwise (the interval contains 0 and reaches the threshold)
    """
    if lo >= delta:
        return "gain"
    if hi <= -delta:
        return "loss"
    if lo > -delta and hi < delta:
        return "negligible"
    if lo > 0:
        return "positive"
    if hi < 0:
        return "negative"
    return "inconclusive"


def bonferroni_probs(k, alpha=ALPHA):
    """Lower and upper percentile probabilities of a simultaneous interval over k contrasts."""
    a = alpha / k
    return a / 2, 1 - a / 2


def n_boot(k):
    """Bootstrap size for a question family of k contrasts: >= 50 replicates per tail."""
    return int(max(20_000, 2_000 * k))


# ---------------------------------------------------------------------------
# Weighted median, vectorised over columns
# ---------------------------------------------------------------------------

def weighted_median_cols(V, W):
    """Weighted median of every column of V with weights W (same shape).

    Rows with weight 0 are ignored. With equal weights this equals np.median,
    including the mean of the two middle values when their count is even.
    """
    V = np.where(W > 0, V, np.inf)          # ignored rows sort last
    order = np.argsort(V, axis=0, kind="mergesort")
    Vs = np.take_along_axis(V, order, axis=0)
    Ws = np.take_along_axis(W, order, axis=0)
    cw = np.cumsum(Ws, axis=0)
    total = cw[-1]
    half = total / 2
    k = (cw < half - 1e-9 * total).sum(axis=0)       # first index with cw >= half
    cols = np.arange(V.shape[1])
    at_half = np.isclose(cw[k, cols], half, rtol=0, atol=1e-9 * total)
    nxt = np.minimum(k + 1, V.shape[0] - 1)
    out = Vs[k, cols].copy()
    use_mid = at_half & np.isfinite(Vs[nxt, cols]) & (nxt > k)
    out[use_mid] = 0.5 * (Vs[k, cols][use_mid] + Vs[nxt, cols][use_mid])
    return out


# ---------------------------------------------------------------------------
# Two-level (hierarchical) bootstrap: training pools, then series
# ---------------------------------------------------------------------------

class HierarchicalBootstrap:
    """Resamples the training pools of each frequency, then series within each pool.

    `inst` has one row per series-feature instance (cohort scope), with columns
    series_id, freq, pool_id, source. Row order defines the instance index used
    by the value matrices passed to `run`.

    Level 1: within each frequency, draw as many pools as there are (6), with
    replacement. Level 2: within every drawn pool and source (M3, M4), draw as
    many instances as there are, with replacement.

    Statistic of a replicate: every distinct series drawn at least once gets the
    mean of its drawn instance values and the weight
    (number of draws) / (number of instances it has in the contrast);
    the statistic is the weighted median (or weighted mean) over series.
    Without resampling every weight is 1, so the statistic equals the published
    estimand: the median over distinct series of the per-series mean.
    """

    def __init__(self, inst: pd.DataFrame):
        need = {"series_id", "freq", "pool_id", "source"}
        missing = need - set(inst.columns)
        if missing:
            raise ValueError(f"missing columns: {missing}")
        inst = inst.reset_index(drop=True)
        self.n_inst = len(inst)
        codes, uniques = pd.factorize(inst["series_id"])
        self.series_codes = codes
        self.series_ids = np.asarray(uniques)
        self.n_series = len(uniques)
        # incidence matrix series x instance
        self.A = sparse.csr_matrix(
            (np.ones(self.n_inst), (codes, np.arange(self.n_inst))),
            shape=(self.n_series, self.n_inst))
        self.groups = []   # per frequency: list of pools, each a list of index arrays per source
        for _, df_f in inst.groupby("freq", sort=True):
            pools = []
            for _, df_p in df_f.groupby("pool_id", sort=True):
                pools.append([g.index.to_numpy() for _, g in df_p.groupby("source", sort=True)])
            self.groups.append(pools)

    def _draw_counts(self, rng):
        parts = []
        for pools in self.groups:
            for p in rng.integers(0, len(pools), len(pools)):
                for idx in pools[p]:
                    parts.append(idx[rng.integers(0, len(idx), len(idx))])
        return np.bincount(np.concatenate(parts), minlength=self.n_inst).astype(float)

    def _stat(self, X0, M, P, counts, stat):
        cM = M * counts[:, None]
        N = self.A @ cM                         # draws per series and contrast
        S = self.A @ (X0 * counts[:, None])
        with np.errstate(invalid="ignore", divide="ignore"):
            V = S / N
            W = np.where(P > 0, N / P, 0.0)
        V = np.nan_to_num(V)
        if stat == "median":
            return weighted_median_cols(V, W)
        if stat == "mean":
            return (V * W).sum(axis=0) / W.sum(axis=0)
        if stat == "winrate":            # share of series whose mean delta is > 0
            return ((V > 0) * W).sum(axis=0) / W.sum(axis=0)
        raise ValueError(stat)

    def point(self, X, stat="median"):
        """Published estimand (no resampling)."""
        X = np.asarray(X, float)
        X = X[:, None] if X.ndim == 1 else X
        M = ~np.isnan(X)
        X0 = np.where(M, X, 0.0)
        P = self.A @ M.astype(float)
        return self._stat(X0, M.astype(float), P, np.ones(self.n_inst), stat)

    def run(self, X, B, stat="median", seed=BOOT_SEED):
        """Replicates (B x C) for the instance value matrix X (n_inst x C, NaN = not in contrast).

        The same seed gives the same pool and series draws for every call, so all
        contrasts share their resamples, and the first B replicates of a longer
        run are those of a shorter run.
        """
        X = np.asarray(X, float)
        X = X[:, None] if X.ndim == 1 else X
        M = ~np.isnan(X)
        X0 = np.where(M, X, 0.0)
        Mf = M.astype(float)
        P = self.A @ Mf
        rows = np.where(P.sum(axis=1) > 0)[0]       # only series that enter some contrast
        A_full, self.A = self.A, self.A[rows]
        try:
            rng = np.random.default_rng(seed)
            out = np.empty((B, X.shape[1]))
            for b in range(B):
                out[b] = self._stat(X0, Mf, P[rows], self._draw_counts(rng), stat)
        finally:
            self.A = A_full
        return out


def series_values_replicates(hb, x, B, seed=BOOT_SEED):
    """Yield (series index, per-series value) for B replicates of one contrast.

    Used for the dose-response band: each replicate gives the per-series means of
    the drawn instances (series not drawn are left out); weights are not used.
    """
    x = np.asarray(x, float)
    m = ~np.isnan(x)
    x0 = np.where(m, x, 0.0)
    rng = np.random.default_rng(seed)
    for _ in range(B):
        c = hb._draw_counts(rng) * m
        n = hb.A @ c
        s = hb.A @ (x0 * c)
        idx = np.where(n > 0)[0]
        yield idx, s[idx] / n[idx]


def seed_bootstrap(values, freq, source, B, stat="median", seed=BOOT_SEED):
    """Two-level bootstrap on the evolving-seasonality sample: seeds, then series.

    values: array (n_series, n_seeds) of per-series deltas, one column per seed.
    Within each frequency, draw n_seeds seeds with replacement; each series gets
    the mean of its values under the drawn seeds; series are resampled within
    frequency and source. Point estimate: per-series mean over the seeds, then
    the median over series. stat: "median", "mean" or "winrate".
    """
    values = np.asarray(values, float)
    freq = np.asarray(freq)
    n, s = values.shape
    rng = np.random.default_rng(seed)
    strata = [np.asarray(ix) for _, ix in sorted(
        pd.DataFrame({"f": freq, "s": np.asarray(source)}).groupby(["f", "s"]).indices.items())]
    freqs = np.unique(freq)
    out = np.empty(B)
    for b in range(B):
        v = np.empty(n)
        for f in freqs:
            seeds = rng.integers(0, s, s)
            in_f = freq == f
            v[in_f] = values[in_f][:, seeds].mean(axis=1)
        w = np.bincount(np.concatenate(
            [ix[rng.integers(0, len(ix), len(ix))] for ix in strata]), minlength=n).astype(float)
        if stat == "median":
            out[b] = weighted_median_cols(v[:, None], w[:, None])[0]
        elif stat == "mean":
            out[b] = (v * w).sum() / w.sum()
        else:
            out[b] = ((v > 0) * w).sum() / w.sum()
    return out


def summarise(point, reps, k, delta=DELTA, alpha=ALPHA):
    """Point estimate, 95% interval, simultaneous interval over k contrasts and label."""
    reps = np.asarray(reps)
    lo95, hi95 = np.quantile(reps, [0.025, 0.975], axis=0)
    pl, pu = bonferroni_probs(k, alpha)
    lo, hi = np.quantile(reps, [pl, pu], axis=0)
    point = np.atleast_1d(point)
    lo, hi, lo95, hi95 = map(np.atleast_1d, (lo, hi, lo95, hi95))
    return pd.DataFrame({
        "estimate": point, "ci95_lo": lo95, "ci95_hi": hi95,
        "sim_lo": lo, "sim_hi": hi,
        "label": [label(a, b, delta) for a, b in zip(lo, hi)],
    })


# ---------------------------------------------------------------------------
# Effect sizes on other scales
# ---------------------------------------------------------------------------

def log_ratio_effect(mae_direct, mae_stl):
    """-log(MAE_STL / MAE_Direct) per forecast (positive: STL better); NaN if either is 0."""
    d = np.asarray(mae_direct, float)
    s = np.asarray(mae_stl, float)
    ok = (d > 0) & (s > 0)
    out = np.full(d.shape, np.nan)
    out[ok] = -np.log(s[ok] / d[ok])
    return out


def mase_delta(mae_direct, mae_stl, mae_naive_test, scale_insample, cap=CAP):
    """Direct minus STL MASE per forecast, with both MAEs capped at cap * naive test MAE."""
    lim = cap * np.asarray(mae_naive_test, float)
    d = np.minimum(np.asarray(mae_direct, float), lim)
    s = np.minimum(np.asarray(mae_stl, float), lim)
    return (d - s) / np.asarray(scale_insample, float)


# ---------------------------------------------------------------------------
# Dose-response curve
# ---------------------------------------------------------------------------

GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)


def lowess_curve(x, y, grid=GRID, frac=0.3):
    """LOESS of the per-series delta on a feature, evaluated on a fixed grid."""
    import statsmodels.api as sm
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    fit = sm.nonparametric.lowess(y, x, frac=frac, it=3, delta=0.005, return_sorted=True)
    xs, inv = np.unique(fit[:, 0], return_inverse=True)        # ties: average the fit
    ys = np.bincount(inv, weights=fit[:, 1]) / np.bincount(inv)
    inside = (grid >= xs[0]) & (grid <= xs[-1])
    out = np.full(grid.shape, np.nan)
    out[inside] = np.interp(grid[inside], xs, ys)
    return out


def upper_crossing(grid, curve, level):
    """Smallest grid value from which the curve stays >= level up to the last defined point.

    NaN if the curve is below level at its last defined point (no crossing).
    """
    ok = ~np.isnan(curve)
    g, c = grid[ok], curve[ok]
    if len(c) == 0 or c[-1] < level:
        return np.nan
    below = np.where(c < level)[0]
    return g[0] if len(below) == 0 else g[below[-1] + 1]
