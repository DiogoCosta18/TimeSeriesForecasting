"""Eligibility and cutoffs (protocol D3, D4, D8; test U2; gate G8; defect M4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.eligibility import (
    NOT_COMPUTED,
    IneligibleSeriesError,
    eligibility_length,
    eligibility_table,
    length_table,
    require_eligible,
)
from src.data.validation import make_cutoffs
from src.features.compute_all import QUALITY_FLAG_NAMES


def frame(lengths: dict[str, int], source: str = "M4_Monthly") -> pd.DataFrame:
    rows = [
        {"unique_id": uid, "source_dataset": source, "t": t, "ds": t, "y": float(t)}
        for uid, n in lengths.items() for t in range(1, n + 1)
    ]
    return pd.DataFrame(rows)


def ok_flags(ids, **override) -> pd.DataFrame:
    rows = [{"unique_id": uid, **{col: "ok" for col in QUALITY_FLAG_NAMES.values()}} for uid in ids]
    df = pd.DataFrame(rows)
    for uid, (col, value) in override.items():
        df.loc[df["unique_id"] == uid, col] = value
    return df


def test_eligibility_length_is_3m_plus_h():
    assert eligibility_length(12, 18) == 54
    assert eligibility_length(4, 8) == 20


@pytest.mark.parametrize("m, h", [(12, 18), (4, 8)])
def test_u2_history_of_exactly_L_is_eligible_and_L_minus_1_is_not(m, h):
    L = eligibility_length(m, h)
    lengths = length_table(frame({"short": L - 1 + 3 * h, "exact": L + 3 * h, "long": L + 3 * h + 40}), h, 3)
    assert dict(zip(lengths["unique_id"], lengths["history_length"])) == {"exact": L, "long": L + 40, "short": L - 1}
    table = eligibility_table(lengths, ok_flags(["exact", "long"]), m, h)
    assert dict(zip(table["unique_id"], table["eligible"])) == {"exact": True, "long": True, "short": False}
    short = table.set_index("unique_id").loc["short"]
    assert short["ineligible_reason"] == f"history_length {L - 1} < L {L}"
    assert (short[list(QUALITY_FLAG_NAMES.values())] == NOT_COMPUTED).all()


def test_d4_any_non_ok_flag_makes_a_series_ineligible_with_its_reason():
    lengths = length_table(frame({"a": 120, "b": 120}), 18, 3)
    flags = ok_flags(["a", "b"], b=("feature_quality_flag_arch_stat", "undefined:constant"))
    table = eligibility_table(lengths, flags, 12, 18).set_index("unique_id")
    assert table.loc["a", "eligible"] and table.loc["a", "ineligible_reason"] == ""
    assert not table.loc["b", "eligible"]
    assert table.loc["b", "ineligible_reason"] == "feature_arch_stat=undefined:constant"


def test_flags_must_cover_exactly_the_length_eligible_series():
    lengths = length_table(frame({"a": 120, "b": 120, "c": 30}), 18, 3)
    with pytest.raises(ValueError, match="exactly once"):
        eligibility_table(lengths, ok_flags(["a"]), 12, 18)
    with pytest.raises(ValueError, match="exactly once"):
        eligibility_table(lengths, ok_flags(["a", "b", "c"]), 12, 18)


def test_g8_no_eligible_series_has_a_history_shorter_than_L():
    rng = np.random.default_rng(0)
    lengths = length_table(frame({f"s{i}": int(n) for i, n in enumerate(rng.integers(60, 200, 50))}), 18, 3)
    computed = lengths.loc[lengths["history_length"] >= 54, "unique_id"]
    table = eligibility_table(lengths, ok_flags(computed), 12, 18)
    assert (table.loc[table["eligible"], "history_length"] >= 54).all()
    assert table["eligible"].sum() == len(computed)


def test_require_eligible_refuses_ineligible_or_unknown_ids():
    lengths = length_table(frame({"a": 120, "b": 40}), 18, 3)
    table = eligibility_table(lengths, ok_flags(["a"]), 12, 18)
    require_eligible(table, ["a"])
    with pytest.raises(IneligibleSeriesError, match="2 ineligible or unknown"):
        require_eligible(table, ["a", "b", "zzz"])


# --- cutoffs (D8, U2) ----------------------------------------------------------------------

@pytest.mark.parametrize("m, h", [(12, 18), (4, 8)])
def test_u2_cutoffs_follow_the_protocol_and_refuse_short_histories(m, h):
    L = eligibility_length(m, h)
    n = L + 3 * h + 5
    cutoffs = make_cutoffs(frame({"a": n}), h, 3, h, min_history=L)
    assert cutoffs["window"].tolist() == [0, 1, 2]
    assert cutoffs["train_end_idx"].tolist() == [n - 3 * h, n - 2 * h, n - h]  # n - (3 - w)h
    assert cutoffs["cutoff"].tolist() == [n - 3 * h, n - 2 * h, n - h]          # ds of the last training point
    make_cutoffs(frame({"exact": L + 3 * h}), h, 3, h, min_history=L)
    with pytest.raises(IneligibleSeriesError, match="1 series have fewer than"):
        make_cutoffs(frame({"exact": L + 3 * h, "short": L - 1 + 3 * h}), h, 3, h, min_history=L)


def test_make_cutoffs_requires_the_minimum_history_explicitly():
    with pytest.raises(TypeError):
        make_cutoffs(frame({"a": 120}), 18, 3, 18)  # noqa: the keyword is mandatory
