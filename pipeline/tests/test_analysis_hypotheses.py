"""H1-H7 on toy instance tables with known effects (protocol Sections 5.2-5.4; D26, D27).

Noise depends only on series, window and model, so a contrast without a planted effect is
exactly zero and a planted effect is recovered exactly: orientation, Holm family sizes,
labels, family weighting and pairwise deletion are checked against known answers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import data as ad
from src.analysis import hypotheses as hy
from src.forecast.registry import FAMILY, LINEARITY, MODELS, STRATEGIES

S0 = 7
BUCKETS = ["Low", "Medium", "High"]


def toy(effect, n_series: int = 36, feature: str = "feature_evolving_seasonality") -> pd.DataFrame:
    """Every instance of one feature sample; RelNaive = 1 + noise(series, window, model) + effect(...)."""
    rng = np.random.default_rng(0)
    noise = rng.uniform(0, 0.5, size=(n_series, 3, len(MODELS)))
    rows = []
    for i in range(n_series):
        frequency = "monthly" if i % 2 == 0 else "quarterly"
        uid, bucket = f"{'M4' if i % 4 < 2 else 'M3'}_{frequency.capitalize()}_S{i:02d}", BUCKETS[i % 3]
        for window in range(3):
            for k, model in enumerate(MODELS):
                fam = FAMILY[model]
                for strategy in STRATEGIES:
                    for scope in (["cohort"] if fam == "statistical" else ["cohort", bucket.lower()]):
                        value = 1 + noise[i, window, k] + effect(fam, model, strategy, scope, bucket)
                        rows.append({"feature_name": feature, "frequency": frequency, "unique_id": uid,
                                     "source_dataset": uid.rsplit("_", 1)[0], "window": window, "family": fam, "model": model,
                                     "strategy": strategy, "scope": scope, "bucket": bucket,
                                     "seed_key": -1 if fam == "statistical" else S0, "relnaive_capped": value})
    inst = pd.DataFrame(rows)
    for c in ad.CATEGORIES:
        inst[c] = inst[c].astype("category")
    return inst


def _row(df, **keys):
    sel = np.ones(len(df), dtype=bool)
    for k, v in keys.items():
        sel &= (df[k].astype(str) == v).to_numpy()
    (i,) = np.flatnonzero(sel)
    return df.iloc[i]


def stl_ml_sn(fam, model, strategy, scope, bucket):
    return -0.05 if (fam == "ml" and strategy == "stl_sn") else 0.0     # STL-SN lowers ML error by 0.05


def test_h1_recovers_a_planted_stl_effect_and_its_orientation():
    res = hy.h1(toy(stl_ml_sn), S0)
    assert len(res) == 8 and (res["n_tests_in_family"] == 8).all()
    ml = _row(res, family="ml", strategy="stl_sn")
    assert ml["median"] == pytest.approx(0.05) and ml["label"] == "gain" and ml["rank_biserial"] == pytest.approx(1.0)
    assert ml["ci_low"] == pytest.approx(0.05) and ml["instance_win_rate"] == 1.0
    others = res[~((res["family"] == "ml") & (res["strategy"] == "stl_sn"))]
    assert (others["label"] == "no evidence").all() and (others["p"] == 1.0).all() and (others["instance_tie_share"] == 1.0).all()


def test_h2_and_h3_orientation():
    inst = toy(stl_ml_sn)
    h2 = hy.h2(inst, S0)
    assert len(h2) == 12
    r = _row(h2, strategy="stl_sn", family_a="statistical", family_b="ml")
    assert r["median"] == pytest.approx(-0.05) and r["label"] == "loss" and r["larger_stl_effect"] == "ml"
    h3 = hy.h3(inst, S0)
    assert len(h3) == 12 and (h3["n_tests_in_family"] == 12).all()
    r = _row(h3, family="ml", first="direct", second="stl_sn")
    assert r["median"] == pytest.approx(0.05) and r["label"] == "gain" and r["favours"] == "stl_sn"
    assert r["series_second_better"] == 36
    r = _row(h3, family="ml", first="stl_sn", second="stl_ac")
    assert r["median"] == pytest.approx(-0.05) and r["favours"] == "stl_sn"


def test_pairwise_deletion_is_counted():
    inst = toy(stl_ml_sn)
    failed = inst.index[(inst["family"] == "statistical") & (inst["strategy"] == "direct")][:1]
    inst.loc[failed, "relnaive_capped"] = np.nan          # a failed statistical Direct row
    res = hy.h1(inst, S0)
    assert _row(res, family="statistical", strategy="stl_sn")["instance_n_excluded"] == 1
    assert _row(res, family="statistical", strategy="stl_ac")["instance_n_excluded"] == 1
    assert _row(res, family="ml", strategy="stl_sn")["instance_n_excluded"] == 0


def specialisation(fam, model, strategy, scope, bucket):
    if scope == "cohort" or fam == "statistical":
        return 0.0
    return -0.02 if LINEARITY[model] == "nonlinear" else 0.03   # tercile helps non-linear, harms linear models


def test_h5_and_h6_use_equal_family_weights():
    inst = toy(specialisation)
    h5 = hy.h5(inst, S0)
    assert len(h5) == 9
    assert _row(h5, family="ml", strategy="direct")["median"] == pytest.approx((0.02 * 2 - 0.03 * 2) / 4)   # negligible
    assert _row(h5, family="ml", strategy="direct")["label"] == "negligible"
    assert _row(h5, family="transformer", strategy="direct")["label"] == "gain"
    h6 = hy.h6(inst, S0)
    assert len(h6) == 3 and (h6["label"] == "gain").all()
    # non-linear: ml, neural, transformer each +0.02; linear: ml and neural each -0.03 -> +0.05
    assert h6["median"].to_numpy() == pytest.approx([0.05] * 3)
    assert h6["median_nonlinear"].to_numpy() == pytest.approx([0.02] * 3)


def trend(fam, model, strategy, scope, bucket):
    return -0.01 * BUCKETS.index(bucket) if strategy == "stl_ac" else 0.0    # STL-AC helps more in higher terciles


def _tables(inst, degenerate=False):
    s = inst[["frequency", "feature_name", "unique_id", "bucket"]].astype(str).drop_duplicates()
    s["feature_value"] = s["bucket"].map({b: k for k, b in enumerate(BUCKETS)}) + s["unique_id"].str[-2:].astype(int) / 100
    summary = pd.DataFrame({"feature_name": "feature_evolving_seasonality", "frequency": ["monthly", "quarterly"],
                            "degenerate": [degenerate, False]})
    return {"samples": s, "bucket_summary": summary}


def test_h4_trend_tests_and_the_d7_exclusion():
    inst = toy(trend)
    res = hy.h4(inst, _tables(inst), S0)
    jt = res[res["test_family"] == "H4a-JT"].set_index("strategy")
    sp = res[res["test_family"] == "H4b-Spearman"].set_index("strategy")
    assert jt.loc["stl_ac", "high_minus_low"] == pytest.approx(0.02) and jt.loc["stl_ac", "label"] == "gain"
    assert jt.loc["stl_ac", "direction"] == "increasing" and jt.loc["stl_ac", "z"] > 0
    assert jt.loc["stl_sn", "label"] == "no evidence"
    assert sp.loc["stl_ac", "rho"] > 0.8 and sp.loc["stl_ac", "label"] == "gain"
    # degenerate monthly: the trend test drops monthly series, Spearman keeps them
    res = hy.h4(inst, _tables(inst, degenerate=True), S0)
    jt = res[res["test_family"] == "H4a-JT"].set_index("strategy")
    assert jt.loc["stl_ac", "n_series"] == 18 and jt.loc["stl_ac", "excluded_frequencies"] == "monthly"
    assert res[res["test_family"] == "H4b-Spearman"]["n_series"].eq(36).all()


def best_config(fam, model, strategy, scope, bucket):
    return -0.5 if (model, strategy) == ("PatchTST", "stl_sn") else 0.0


def test_h7_ranks_lowest_error_first_with_36_and_27_configurations():
    inst = toy(best_config)
    cohort = hy.h7(inst, "cohort")
    assert cohort["summary"]["k"] == 36 and cohort["summary"]["n_series"] == 36
    assert cohort["ranks"].iloc[0]["configuration"] == "PatchTST|stl_sn" and cohort["ranks"].iloc[0]["mean_rank"] == 1.0
    assert len(cohort["pairs"]) == 36 * 35 // 2
    tercile = hy.h7(inst, "tercile", "monthly")
    assert tercile["summary"]["k"] == 27 and tercile["summary"]["n_series"] == 18
    pair = tercile["pairs"][(tercile["pairs"]["first"] == "PatchTST|direct") & (tercile["pairs"]["second"] == "PatchTST|stl_sn")].iloc[0]
    assert pair["median_relnaive_difference"] == pytest.approx(0.5) and bool(pair["practically_meaningful"]) == bool(pair["separated"])


def test_statistical_rows_enter_every_feature_sample_with_its_bucket():
    """D18: one statistical forecast per series serves every feature sample holding it."""
    base = {"task_id": "t", "frequency": "monthly", "source_dataset": "M4_Monthly", "window": 0, "strategy": "direct",
            "scope": "cohort", "status": "trained", **{v: 1.0 for v in ad.VALUES}}
    rows = pd.DataFrame([
        {**base, "unique_id": "A", "family": "statistical", "model": "ETS", "feature_name": None, "bucket": None, "seed": pd.NA},
        {**base, "unique_id": "A", "family": "ml", "model": "Ridge", "feature_name": "f1", "bucket": "Low", "seed": S0},
        {**base, "unique_id": "A", "family": "ml", "model": "Ridge", "feature_name": "f1", "bucket": "Low", "seed": S0 + 1},
    ])
    buckets = pd.DataFrame({"frequency": "monthly", "feature_name": ["f1", "f2"], "unique_id": "A",
                            "feature_tercile_bucket": ["Low", "High"]})
    inst = ad.build_instances(rows, rows.iloc[:0], buckets, S0)
    stat = inst[inst["family"] == "statistical"].astype({"feature_name": str, "bucket": str})
    assert sorted(zip(stat["feature_name"], stat["bucket"])) == [("f1", "Low"), ("f2", "High")]
    assert inst["main_seed"].tolist().count(True) == 3 and (inst.loc[inst["seed_key"] == S0 + 1, "main_seed"] == False).all()  # noqa: E712
    with pytest.raises(ad.AnalysisError, match="outside every feature sample"):
        ad.build_instances(rows, rows.iloc[:0], buckets[buckets["unique_id"] == "B"], S0)


def test_a1_standing_survives_a_configuration_whose_instances_all_failed():
    from src.analysis.secondary import a1_standing

    inst = toy(stl_ml_sn)
    inst["mase"] = inst["smape"] = inst["pocid"] = 1.0
    inst.loc[(inst["model"] == "ETS") & (inst["strategy"] == "stl_ac"), "relnaive_capped"] = np.nan
    standing, top = a1_standing(inst)
    assert _row(standing, model="ETS", strategy="stl_ac")["n_failed"] == _row(standing, model="ETS", strategy="stl_ac")["n"]
    assert np.isnan(_row(top, model="ETS", strategy="stl_ac")["relnaive_mean"])
    assert top["relnaive_mean"].notna().sum() == len(top) - 1
