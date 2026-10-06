"""Exploratory analyses E1-E3 on hand-built instances whose answers are worked out by hand."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import exploratory as ex
from src.analysis.data import AnalysisError

# (series, frequency, family, model, window, direct, stl_sn, stl_ac)
CELLS = [
    ("A", "monthly", "ml", "m1", 0, 1.0, 0.8, 0.9),        # best STL-SN, gain 0.2
    ("A", "monthly", "ml", "m2", 0, 0.5, 0.5, 0.7),        # tie Direct / STL-SN -> Direct, gain 0
    ("B", "quarterly", "ml", "m1", 0, 0.6, 0.7, 0.4),      # best STL-AC, gain 0.2
    ("B", "quarterly", "ml", "m2", 0, 0.9, 0.85, 1.0),     # two windows: means 0.8 / 0.75 / 0.9
    ("B", "quarterly", "ml", "m2", 1, 0.7, 0.65, 0.8),     #   -> best STL-SN, gain 0.05
    ("C", "monthly", "ml", "m1", 0, 0.5, 0.4, np.nan),     # STL-AC failed: pairwise deletion
    ("A", "monthly", "statistical", "s1", 0, 0.9, 1.0, 1.1),  # best Direct, gain 0
]
POCID = {"direct": 0.6, "stl_sn": 0.5, "stl_ac": 0.4}


def _instances() -> pd.DataFrame:
    rows = []
    for uid, freq, fam, model, window, *values in CELLS:
        for strategy, v in zip(["direct", "stl_sn", "stl_ac"], values):
            rows.append({"unique_id": uid, "frequency": freq, "family": fam, "model": model, "window": window,
                         "strategy": strategy, "scope": "cohort", "main_seed": True, "relnaive_capped": v,
                         "pocid": POCID[strategy]})
    # ignored: a tercile-scope row and a row of an extra seed
    rows.append({**rows[0], "scope": "low", "relnaive_capped": 99.0})
    rows.append({**rows[0], "main_seed": False, "relnaive_capped": 99.0})
    inst = pd.DataFrame(rows)
    for c in ("unique_id", "frequency", "family", "model", "strategy", "scope"):
        inst[c] = inst[c].astype("category")
    return inst


def test_series_by_strategy_averages_windows_and_deletes_pairwise():
    t = ex.series_by_strategy(_instances()).set_index(["unique_id", "model"])
    assert len(t) == 5 and ("C", "m1") not in t.index
    assert t.loc[("B", "m2"), ["direct", "stl_sn", "stl_ac"]].tolist() == pytest.approx([0.8, 0.75, 0.9])
    assert t.loc[("A", "m1"), "direct"] == pytest.approx(1.0)          # tercile and extra-seed rows ignored


def test_oracle_shares_and_gains():
    by_family, by_model = ex.oracle(ex.series_by_strategy(_instances()))
    ml = by_family.set_index("family").loc["ml"]
    assert (ml.n_series, ml.n_series_models) == (2, 4)
    assert [ml.best_share_direct, ml.best_share_stl_sn, ml.best_share_stl_ac] == pytest.approx([0.25, 0.5, 0.25])
    assert ml.gain_median == pytest.approx(0.1125) and ml.gain_mean == pytest.approx(0.1125)  # series A 0.1, B 0.125
    assert ml["share_series_gain_ge_0.01"] == pytest.approx(1.0)
    stat = by_family.set_index("family").loc["statistical"]
    assert stat.best_share_direct == pytest.approx(1.0) and stat.gain_median == pytest.approx(0.0)
    assert by_family["family"].tolist() == ["statistical", "ml"]
    m2 = by_model.set_index(["family", "model"]).loc[("ml", "m2")]
    assert [m2.best_share_direct, m2.best_share_stl_sn] == pytest.approx([0.5, 0.5])


def test_evolving_seasonality_rule():
    features = pd.DataFrame({"unique_id": ["A", "B", "C"], "frequency": ["monthly", "quarterly", "monthly"],
                             "feature_evolving_seasonality": [0.10, 0.30, 0.05]})
    cuts = pd.DataFrame({"feature_name": ["feature_evolving_seasonality"] * 2 + ["feature_arch_stat"],
                         "frequency": ["monthly", "quarterly", "monthly"], "tercile_cut_1": [0.13, 0.14, 99.0]})
    t = ex.feature_rule(ex.series_by_strategy(_instances()), features, cuts).set_index("family").loc["ml"]
    # A decomposes (0.10 < 0.13): STL-SN gains 0.2 and 0.0 -> 0.1; B stays Direct (0.30 > 0.14) -> 0
    assert t.share_decomposed == pytest.approx(0.5)
    assert (t.rule_median, t.rule_mean, t.rule_share_improved) == pytest.approx((0.05, 0.05, 0.5))
    # always STL-SN: A 0.1, B mean(-0.1, 0.05) = -0.025
    assert (t.always_stl_sn_median, t.always_stl_sn_share_improved) == pytest.approx((0.0375, 0.5))
    assert t.oracle_mean == pytest.approx(0.1125)
    with pytest.raises(AnalysisError, match="without feature_evolving_seasonality"):
        ex.feature_rule(ex.series_by_strategy(_instances()), features.iloc[1:], cuts)


def test_pocid_table():
    p = ex.pocid_table(_instances()).set_index(["model", "strategy"])
    assert p.loc[("m2", "direct"), "n"] == 3                            # A window 0, B windows 0 and 1
    assert p.loc[("m2", "direct"), "relnaive_mean"] == pytest.approx((0.5 + 0.9 + 0.7) / 3)
    assert p.loc[("m1", "stl_ac"), "pocid_mean"] == pytest.approx(0.4)


def test_outputs_are_never_overwritten(tmp_path):
    (tmp_path / "old.csv").write_text("x")
    with pytest.raises(AnalysisError, match="not empty"):
        ex.run_exploratory(tmp_path / "run", tmp_path)
