"""Exploratory analyses E7-E9 (change log v2.4) on small inputs with answers worked out by hand."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from src.analysis import exploratory2 as ex2
from src.analysis import hypotheses as hy
from src.metrics.rel_naive import rel_naive


def brute_hodges_lehmann(x):
    x = np.asarray(x, dtype=float)
    return float(np.median([(x[i] + x[j]) / 2 for i, j in itertools.combinations_with_replacement(range(len(x)), 2)]))


@pytest.mark.parametrize("seed", range(12))
def test_hodges_lehmann_matches_the_median_of_walsh_averages(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 41))
    x = np.round(rng.standard_t(3, n), 1 if seed % 2 else 6)          # odd seeds: many ties and zeros
    assert ex2.hodges_lehmann(x) == pytest.approx(brute_hodges_lehmann(x), abs=1e-12)


def test_hodges_lehmann_and_median_can_disagree_in_sign():
    x = np.array([-0.01] * 6 + [0.5] * 5)      # most series lose a little, a minority gains a lot
    assert np.median(x) < 0 < ex2.hodges_lehmann(x)


def test_sign_test_is_a_binomial_test_without_zeros():
    out = ex2.sign_test([1.0, 2.0, -1.0, 0.0, 3.0])
    assert (out["n_positive"], out["n_negative"]) == (3, 1)
    assert out["p_sign"] == pytest.approx(sps.binomtest(3, 4, 0.5).pvalue)
    assert ex2.sign_test([0.0, 0.0])["p_sign"] == 1.0


def test_capture_records_what_the_wilcoxon_test_would_see_and_restores_it():
    ps = pd.DataFrame({"unique_id": list("abcdef"), "g": ["x", "x", "x", "y", "y", "y"], "delta": [1.0, -2.0, 3.0, 0.5, 0.25, -0.1]})
    store = {}
    original = hy._test
    with ex2.capture_tests(store):
        hy.one_sample(ps, ["g"], 1, "H9")
    assert hy._test is original
    assert set(store) == {"H9|x", "H9|y"} and store["H9|x"].tolist() == [1.0, -2.0, 3.0]


def test_e8_table_holm_per_family_and_flags_label_changes():
    rng = np.random.default_rng(3)
    skewed = np.r_[np.full(300, -0.02), np.full(200, 0.30)]        # median a loss, Hodges-Lehmann a gain
    series = {"H1|a|stl_sn": rng.normal(-0.1, 0.05, 400), "H1|b|stl_sn": rng.normal(0.0, 0.05, 400), "S1|ETS|log": skewed}
    t = ex2.e8_table(series).set_index(["test_family", "contrast"])
    assert t.loc[("H1", "a|stl_sn"), "n_tests_in_family"] == 2 and t.loc[("S1", "ETS|log"), "n_tests_in_family"] == 1
    assert t.loc[("H1", "a|stl_sn"), "label"] == "loss" and not t.loc[("H1", "a|stl_sn"), "differs"]
    s = t.loc[("S1", "ETS|log")]
    assert s["label_sign_median"] == "loss" and s["label_wilcoxon_hl"] == "gain" and s["differs"] and s["signs_disagree"]


def test_average_pair_scores_the_mean_of_the_two_forecasts():
    y, naive = [10.0, 12.0, 11.0], [9.0, 9.0, 9.0]
    direct = pd.DataFrame({"unique_id": ["s"], "window": [0], "y_test": [y], "yhat": [[11.0, 13.0, 10.0]], "yhat_naive": [naive]})
    stl_sn = pd.DataFrame({"unique_id": ["s"], "window": [0], "yhat": [[9.0, 12.0, 12.0]]})
    r = ex2.average_pair(direct, stl_sn).iloc[0]
    assert r["rn_average"] == pytest.approx(rel_naive(y, [10.0, 12.5, 11.0], naive, clip=ex2.CAP))
    assert r["rn_direct"] == pytest.approx(rel_naive(y, [11.0, 13.0, 10.0], naive, clip=ex2.CAP))
    assert r["rn_raw_stl_sn"] == pytest.approx(rel_naive(y, [9.0, 12.0, 12.0], naive))


def test_cross_table_and_cell_tests():
    rows = []
    for i in range(60):
        ev, st = ex2.TERCILES[i % 3], ex2.TERCILES[(i // 3) % 3]
        gain = 0.2 if (ev == "Low" and st == "High") else -0.05
        rows.append({"unique_id": f"s{i}", "frequency": "monthly", "family": "pooled", "strategy": "stl_sn",
                     "delta": gain + 0.001 * i, "evolving_tercile": ev, "strength_tercile": st})
    d = pd.DataFrame(rows)
    cross = ex2.cross_table(d)
    cell = cross[(cross["frequency"] == "all") & (cross["evolving_tercile"] == "Low") & (cross["strength_tercile"] == "High")].iloc[0]
    # i % 3 == 0 (Low evolving) and (i // 3) % 3 == 2 (High strength): i // 3 in {2, 5, 8, 11, 14, 17}
    assert cell["n_series"] == 6 and cell["median"] > 0.2 and cell["share_positive"] == 1.0
    tests = ex2.cell_tests(d, 1).set_index("cell")
    assert tests.loc["stable and strong", "n_series"] == 6 and tests.loc["stable", "n_series"] == 20
    assert tests.loc["stable", "median"] < 0 < tests.loc["stable and strong", "median"]


# --- end to end, on the synthetic gated run of the S2 tests with S1 and S2 evaluated ------------------------

from test_stage_sensitivity2 import merged, run  # noqa: E402,F401  (fixtures)


def test_run_exploratory2_end_to_end(run):
    from src.stages import sensitivity as s1
    from src.stages import sensitivity2 as s2
    from test_stage_sensitivity import SMALL

    run_dir, data_dir, manifest = run
    for study in (s1.S1, s2.S2):
        for group in ("cpu", "gpu"):
            s1.run_tune(run_dir, SMALL, data_dir, group, manifest, study=study)
        s1.run_freeze(run_dir, study=study)
        for group in ("cpu", "gpu"):
            s1.run_evaluate(run_dir, SMALL, data_dir, group, manifest, study=study)
    out = run_dir / "exploratory2"
    result = ex2.run_exploratory2(run_dir, SMALL, data_dir, out, manifest)
    assert result["tables"] == 10
    e8 = pd.read_csv(out / "tables" / "e8_tests.csv")
    assert set(e8["test_family"]) == set(ex2.E8_FAMILIES)
    sizes = e8.groupby("test_family").size()
    assert (sizes["H1"], sizes["H3"], sizes["S1"], sizes["S2-vs-direct"], sizes["S2-vs-run-stl-sn"]) == (8, 12, 8, 10, 10)
    strength = pd.read_csv(out / "tables" / "e7_strength.csv")
    assert strength["seasonal_strength"].between(0, 1).all() and len(strength) > 0
    e9 = pd.read_csv(out / "tables" / "e9_average_by_family.csv")
    assert list(e9["family"]) == ["statistical", "ml", "neural", "transformer"]
    with pytest.raises(Exception, match="never overwritten"):
        ex2.run_exploratory2(run_dir, SMALL, data_dir, out, manifest)
