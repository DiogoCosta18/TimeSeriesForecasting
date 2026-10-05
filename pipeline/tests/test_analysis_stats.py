"""U19: the statistical procedures against reference implementations and hand-computed cases."""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from src.analysis import stats


def test_holm_matches_statsmodels_and_a_hand_case():
    p = [0.01, 0.04, 0.03, 0.005]
    # by hand: sorted 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    assert np.allclose(stats.holm(p), [0.03, 0.06, 0.06, 0.02])
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(0, 0.2, size=rng.integers(1, 15))
        assert np.allclose(stats.holm(q), multipletests(q, method="holm")[1])
    with pytest.raises(ValueError):
        stats.holm([0.1, np.nan])


def test_signed_rank_matches_scipy_and_rank_biserial_by_hand():
    d = [1, 2, 3, -4]
    # ranks of |d|: 1 2 3 4; R+ = 6, R- = 4 -> r = 2 / 10
    assert stats.rank_biserial(d) == pytest.approx(0.2)
    assert stats.rank_biserial([0, 1, -1]) == pytest.approx(0.0)
    x = np.array([1.0, 2, 3, -4, 5, 6, -7, 8, 0, 0])
    res = stats.signed_rank(x)
    ref = sps.wilcoxon(x, zero_method="wilcox")
    assert (res["n_series"], res["n_nonzero"]) == (10, 8)
    assert res["statistic"] == pytest.approx(ref.statistic) and res["p"] == pytest.approx(ref.pvalue)
    assert res["p"] == pytest.approx(0.3828125)  # exact: W- = 11 of n = 8
    big = np.random.default_rng(1).normal(0.1, 1, 500)
    assert stats.signed_rank(big)["p"] == pytest.approx(sps.wilcoxon(big).pvalue)
    assert stats.signed_rank([0.0, 0.0])["p"] == 1.0
    with pytest.raises(ValueError):
        stats.signed_rank([1.0, np.nan])


def test_d26_labels():
    assert stats.d26_label(0.001, 0.02) == "gain"
    assert stats.d26_label(0.001, -0.01) == "loss"          # |0.01| meets the threshold
    assert stats.d26_label(0.001, 0.0099) == "negligible"
    assert stats.d26_label(0.05, 0.5) == "no evidence"      # not below alpha
    assert stats.d26_label(0.2, -0.3) == "no evidence"


def _exact_jt(groups):
    """Exact null mean and variance of J by enumerating every assignment of values to groups."""
    values = np.concatenate(groups)
    sizes = [len(g) for g in groups]
    js = []
    for perm in set(itertools.permutations(range(len(values)))):
        parts, start = [], 0
        for s in sizes:
            parts.append(values[list(perm[start:start + s])])
            start += s
        js.append(stats.jonckheere_terpstra(parts)["statistic"])
    return np.mean(js), np.var(js)


@pytest.mark.parametrize("groups", [
    [[1.0, 2.0], [3.0, 4.0], [5.0]],                 # no ties
    [[1.0, 2.0], [2.0, 3.0], [3.0, 3.0]],            # ties within and across groups
    [[0.0, 0.0, 1.0], [0.0, 2.0], [1.0, 2.0]],
])
def test_jonckheere_terpstra_moments_match_exact_enumeration(groups):
    exact_mean, exact_var = _exact_jt([np.array(g) for g in groups])
    res = stats.jonckheere_terpstra(groups)
    assert res["mean"] == pytest.approx(exact_mean) and res["variance"] == pytest.approx(exact_var)


def test_jonckheere_terpstra_statistic_by_hand_and_direction():
    # pairs (1,2): 1<3,1<4,2<3,2<4 = 4; (1,3): 2; (2,3): 2 -> J = 8 = maximum
    res = stats.jonckheere_terpstra([[1, 2], [3, 4], [5]])
    assert res["statistic"] == 8 and res["z"] > 0
    assert stats.jonckheere_terpstra([[5], [3, 4], [1, 2]])["z"] < 0
    rng = np.random.default_rng(2)
    trend = [rng.normal(mu, 1, 200) for mu in (0, 0.3, 0.6)]
    assert stats.jonckheere_terpstra(trend)["p"] < 1e-4
    # empty Medium (degenerate bucket): two groups, J equals the Mann-Whitney U of High over Low
    low, high = rng.normal(0, 1, 30), rng.normal(0.5, 1, 40)
    u = sps.mannwhitneyu(high, low).statistic
    assert stats.jonckheere_terpstra([low, [], high])["statistic"] == pytest.approx(u)


def test_friedman_matches_scipy_and_nemenyi_matches_demsar_table():
    rng = np.random.default_rng(3)
    m = rng.normal(size=(40, 5)) + np.array([0, 0.2, 0.4, 0.6, 0.8])
    res = stats.friedman(m)
    ref = sps.friedmanchisquare(*m.T)
    assert res["statistic"] == pytest.approx(ref.statistic) and res["p"] == pytest.approx(ref.pvalue)
    assert res["mean_ranks"].sum() == pytest.approx(5 * 6 / 2)
    assert stats.friedman([[1, 2, 3], [1, 2, 3], [2, 1, 3]])["mean_ranks"] == pytest.approx([4 / 3, 5 / 3, 3])
    # Demsar (2006), Table 5(a): q_0.05 for k = 2..10
    for k, q in zip(range(2, 11), [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164]):
        assert stats.nemenyi_q(k) == pytest.approx(q, abs=1.5e-3)
    assert stats.nemenyi_cd(4, 14) == pytest.approx(2.569 * np.sqrt(4 * 5 / (6 * 14)), rel=1e-3)
    with pytest.raises(ValueError):
        stats.friedman([[1, 2, np.nan], [1, 2, 3]])


def test_bootstrap_is_reproducible_per_contrast_and_covers_the_median():
    x = np.random.default_rng(4).normal(0.3, 1, 300)
    a, b = stats.median_ci(x, 7, "H1|ml|stl_sn"), stats.median_ci(x, 7, "H1|ml|stl_sn")
    assert a == b and stats.median_ci(x, 7, "H1|ml|stl_ac") != a
    assert a[0] < np.median(x) < a[1]
    s = stats.spearman(x, x + np.random.default_rng(5).normal(0, 1, 300), 7, "H4")
    assert s["rho_ci_low"] < s["rho"] < s["rho_ci_high"] and s["p"] < 1e-6
    assert s["rho"] == pytest.approx(sps.spearmanr(x, x + np.random.default_rng(5).normal(0, 1, 300)).statistic)
    flat = stats.spearman(x, np.zeros(300), 7, "H4")   # a constant delta: undefined, no evidence
    assert np.isnan(flat["rho"]) and flat["p"] == 1.0


def test_descriptives_and_weighted_median():
    d = stats.describe([1.0, -1.0, 0.0, 2.0])
    assert (d["n"], d["mean"], d["median"], d["win_rate"], d["tie_share"]) == (4, 0.5, 0.5, 0.5, 0.25)
    assert stats.weighted_median([1, 2, 3, 4], [1, 1, 1, 1]) == np.median([1, 2, 3, 4])
    assert stats.weighted_median([1, 2, 3], [1, 1, 10]) == 3
    # equal family weights: a 1-instance family counts as much as a 3-instance one
    pooled = stats.describe([1.0, 1.0, 1.0, -1.0], weights=[1 / 3, 1 / 3, 1 / 3, 1.0])
    assert pooled["mean"] == pytest.approx(0.0) and pooled["win_rate"] == pytest.approx(0.5)
