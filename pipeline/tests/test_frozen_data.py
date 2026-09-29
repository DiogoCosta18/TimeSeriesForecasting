"""Frozen-data layer: validation rules, content hash, verified loading (protocol D2, test U18).

All data here is synthetic and built inside the tests; the pipeline itself never
generates data.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helpers_frozen import make_canonical, spec, write_frozen
from src.data.freeze_data import (
    MCOMP_SHA256,
    _canonicalize,
    M3C_SHA256,
    _copy_pinned,
    _fetch,
    _m3_id,
    compare_with_copy,
    freeze,
    m3_frame_from_m3c,
    m3_frame_from_mcomp,
    monash_by_id,
    read_m4_official,
    read_monash_tsf,
)
from src.data.frozen import (
    COLUMNS,
    FrozenDataError,
    content_sha256,
    load_frequency,
    load_manifest,
    load_source,
    sha256_file,
    validate_canonical,
    verify_all,
)


# --- content hash ------------------------------------------------------------------

def test_content_hash_ignores_row_order():
    df = make_canonical()
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    assert content_sha256(df) == content_sha256(shuffled)


def test_content_hash_detects_one_ulp_change():
    df = make_canonical()
    altered = df.copy()
    altered.loc[3, "y"] = np.nextafter(altered.loc[3, "y"], np.inf)
    assert content_sha256(df) != content_sha256(altered)


def test_content_hash_detects_id_and_position_changes():
    df = make_canonical()
    renamed = df.copy()
    renamed.loc[renamed["unique_id"] == "M3_Monthly_S1", "unique_id"] = "M3_Monthly_S9"
    swapped = df.copy()
    swapped.loc[[0, 1], "y"] = swapped.loc[[1, 0], "y"].to_numpy()
    assert content_sha256(df) != content_sha256(renamed)
    assert content_sha256(df) != content_sha256(swapped)


# --- validation rules ----------------------------------------------------------------

def test_validate_accepts_a_valid_frame():
    df = make_canonical()
    stats = validate_canonical(df, **spec(df))
    assert stats["n_series"] == 3 and stats["rows"] == 18
    assert (stats["length_min"], stats["length_max"]) == (5, 7)


def _duplicate_t(df):
    df.loc[1, "t"] = 1
    return df


def _t_from_zero(df):
    df.loc[df["unique_id"] == "M3_Monthly_S1", "t"] -= 1
    return df


def _gap_in_t(df):
    df.loc[2, "t"] = 99
    return df


def _nan_y(df):
    df.loc[4, "y"] = np.nan
    return df


def _inf_y(df):
    df.loc[4, "y"] = np.inf
    return df


def _integer_y(df):
    return df.astype({"y": "int64"})


def _reordered_columns(df):
    return df[list(reversed(COLUMNS))]


def _foreign_id(df):
    df.loc[df["unique_id"] == "M3_Monthly_S1", "unique_id"] = "M4_Monthly_S1"
    return df


def _mixed_frequency(df):
    df.loc[0, "frequency"] = "quarterly"
    return df


@pytest.mark.parametrize(
    "mutate",
    [_duplicate_t, _t_from_zero, _gap_in_t, _nan_y, _inf_y, _integer_y, _reordered_columns, _foreign_id, _mixed_frequency],
)
def test_validate_rejects_broken_frames(mutate):
    df = make_canonical()
    expected = spec(df)
    with pytest.raises(FrozenDataError):
        validate_canonical(mutate(df.copy()), **expected)


def test_validate_rejects_wrong_series_count_and_min_length():
    df = make_canonical()
    with pytest.raises(FrozenDataError, match="series"):
        validate_canonical(df, **{**spec(df), "expected_series": 4})
    with pytest.raises(FrozenDataError, match="shortest series"):
        validate_canonical(df, **{**spec(df), "expected_min_length": 4})


# --- verified loading (U18) ------------------------------------------------------------

def test_load_round_trip_and_verify_all(tmp_path):
    m3 = make_canonical("M3_Monthly", seed=1)
    m4 = make_canonical("M4_Monthly", lengths=(8, 5), seed=2)
    frozen_dir, manifest_path = write_frozen(tmp_path, {"M3_Monthly": m3, "M4_Monthly": m4})
    manifest = load_manifest(manifest_path)
    pd.testing.assert_frame_equal(load_source(frozen_dir, "M3_Monthly", manifest), m3)
    both = load_frequency(frozen_dir, "monthly", manifest)
    assert both["unique_id"].nunique() == 5 and len(both) == len(m3) + len(m4)
    assert set(verify_all(frozen_dir, manifest)) == {"M3_Monthly", "M4_Monthly"}


def test_load_refuses_a_single_altered_byte(tmp_path):
    frozen_dir, manifest_path = write_frozen(tmp_path, {"M3_Monthly": make_canonical()})
    path = frozen_dir / "M3_Monthly.parquet"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(FrozenDataError, match="checksum mismatch"):
        load_source(frozen_dir, "M3_Monthly", load_manifest(manifest_path))


def test_load_refuses_missing_file_and_unlisted_source(tmp_path):
    frozen_dir, manifest_path = write_frozen(tmp_path, {"M3_Monthly": make_canonical()})
    manifest = load_manifest(manifest_path)
    with pytest.raises(FrozenDataError, match="not listed"):
        load_source(frozen_dir, "M4_Monthly", manifest)
    (frozen_dir / "M3_Monthly.parquet").unlink()
    with pytest.raises(FrozenDataError, match="missing"):
        load_source(frozen_dir, "M3_Monthly", manifest)


def test_load_refuses_manifest_disagreements(tmp_path):
    frozen_dir, manifest_path = write_frozen(tmp_path, {"M3_Monthly": make_canonical()})
    manifest = load_manifest(manifest_path)
    wrong_rows = json.loads(json.dumps(manifest))
    wrong_rows["frozen_files"]["M3_Monthly"]["rows"] += 1
    with pytest.raises(FrozenDataError, match="rows"):
        load_source(frozen_dir, "M3_Monthly", wrong_rows)
    wrong_content = json.loads(json.dumps(manifest))
    wrong_content["frozen_files"]["M3_Monthly"]["content_sha256"] = "0" * 64
    with pytest.raises(FrozenDataError, match="content hash"):
        verify_all(frozen_dir, wrong_content)


def test_manifest_with_unknown_format_is_refused(tmp_path):
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps({"format": "something-else", "frozen_files": {"x": {}}}), encoding="utf-8")
    with pytest.raises(FrozenDataError, match="format"):
        load_manifest(path)


# --- M4 reader over the official wide files ------------------------------------------------

def _m4_raw(tmp_path: Path, train_rows, test_rows) -> Path:
    """Write wide M4-style files (id column, then values, NaN padding on the right)."""
    base = tmp_path / "raw" / "m4" / "datasets"
    base.mkdir(parents=True)
    width = max(len(r) for _, r in train_rows)
    train = pd.DataFrame([[sid] + list(r) + [np.nan] * (width - len(r)) for sid, r in train_rows],
                         columns=["V1"] + [f"V{i}" for i in range(2, width + 2)])
    test_width = max(len(r) for _, r in test_rows)
    test = pd.DataFrame([[sid] + list(r) + [np.nan] * (test_width - len(r)) for sid, r in test_rows],
                        columns=["V1"] + [f"V{i}" for i in range(2, test_width + 2)])
    train.to_csv(base / "Monthly-train.csv", index=False)
    test.to_csv(base / "Monthly-test.csv", index=False)
    return tmp_path / "raw"


def test_m4_reader_is_identical_to_datasetsforecast_m4_load(tmp_path):
    from datasetsforecast.m4 import M4

    rng = np.random.default_rng(3)
    lengths = {"M1": 5, "M2": 9, "M3": 3, "M10": 7}
    train_rows = [(sid, list(rng.uniform(100, 10000, size=n))) for sid, n in lengths.items()]
    test_rows = [(sid, list(rng.uniform(100, 10000, size=2))) for sid in reversed(lengths)]
    raw = _m4_raw(tmp_path, train_rows, test_rows)
    base = raw / "m4" / "datasets"
    pd.DataFrame({"M4id": list(lengths), "category": "Macro"}).to_csv(base / "M4-info.csv", index=False)
    (base / "submission-Naive2.zip").write_bytes(b"")  # present, so M4.load downloads nothing

    reference, *_ = M4.load(directory=str(raw), group="Monthly", cache=False)
    reference = reference.assign(unique_id=reference["unique_id"].astype(str), ds=reference["ds"].astype("int64"))
    ours = read_m4_official(raw, "Monthly", horizon=2)
    key = ["unique_id", "ds"]
    pd.testing.assert_frame_equal(
        ours.sort_values(key).reset_index(drop=True)[["unique_id", "ds", "y"]],
        reference.sort_values(key).reset_index(drop=True)[["unique_id", "ds", "y"]],
        check_exact=True,
    )

    canonical = _canonicalize(ours, "M4", "Monthly")
    validate_canonical(canonical, source_dataset="M4_Monthly", frequency="monthly", expected_series=4, expected_min_length=5)
    m10 = canonical[canonical["unique_id"] == "M4_Monthly_M10"]
    assert m10["y"].tolist() == train_rows[3][1] + test_rows[0][1]
    assert m10["t"].tolist() == list(range(1, 10)) and m10["ds_source"].tolist() == [str(t) for t in range(1, 10)]


@pytest.mark.parametrize(
    "train_rows, test_rows, message",
    [
        ([("M1", [1.0, np.nan, 3.0])], [("M1", [9.0, 8.0])], "gap-free prefix"),
        ([("M1", [1.0, 2.0])], [("M1", [9.0, 8.0, 7.0])], "exactly 2"),
        ([("M1", [1.0, 2.0]), ("M2", [3.0, 4.0])], [("M1", [9.0, 8.0]), ("M2", [7.0, np.nan])], "exactly 2"),
        ([("M1", [1.0, 2.0]), ("M1", [3.0, 4.0])], [("M1", [9.0, 8.0])], "duplicated id in the training"),
        ([("M1", [1.0, 2.0])], [("M1", [9.0, 8.0]), ("M1", [7.0, 6.0])], "duplicated ids in the test"),
        ([("M1", [1.0, 2.0]), ("M2", [3.0, 4.0])], [("M1", [9.0, 8.0])], "no row in the test file"),
        ([("M1", [1.0, 2.0])], [("M1", [9.0, 8.0]), ("M2", [7.0, 6.0])], "without a training row"),
    ],
)
def test_m4_reader_refuses_what_m4_load_would_pass_over(tmp_path, train_rows, test_rows, message):
    raw = _m4_raw(tmp_path, train_rows, test_rows)
    with pytest.raises(FrozenDataError, match=message):
        read_m4_official(raw, "Monthly", horizon=2)


# --- M3: the original M3C.xls; Monash and Mcomp as cross-checks ----------------------------

def _mcomp(sn, st, x, xx, period="MONTHLY", freq=12, start=Fraction(1990), n=None, h=None):
    """One series shaped like rdata's conversion of an Mcomp ``Mdata`` object."""
    times = [start + Fraction(k, freq) for k in range(len(x) + len(xx))]
    return {
        "sn": np.array([sn]), "st": np.array([st]), "period": np.array([period]), "type": np.array(["MICRO"]),
        "n": np.array([len(x) if n is None else n]), "h": np.array([float(len(xx) if h is None else h)]),
        "x": pd.Series(x, index=times[:len(x)], dtype=float), "xx": pd.Series(xx, index=times[len(x):], dtype=float),
    }


def _mcomp_set(**changes):
    series = {
        "N0010": _mcomp("N0010", "M2", [5.0, 6.0, 7.0], [8.0, 9.0], start=Fraction(1985) + Fraction(10, 12)),
        "N0002": _mcomp("N0002", "M1", [1.5, 2.5], [3.5, 4.5]),
        "N0001": _mcomp("N0001", "Q1", [1.0, 2.0], [3.0, 4.0], period="QUARTERLY", freq=4),
    }
    series.update(changes)
    return series


def test_mcomp_copy_official_ids_positions_and_period_labels():
    frame = m3_frame_from_mcomp(_mcomp_set(), "Monthly", horizon=2)
    assert frame["unique_id"].tolist() == ["N0002"] * 4 + ["N0010"] * 5
    assert frame["ds"].tolist() == [1, 2, 3, 4, 1, 2, 3, 4, 5]
    assert frame["y"].tolist() == [1.5, 2.5, 3.5, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert frame["ds_source"].tolist()[4:] == ["1985-11", "1985-12", "1986-01", "1986-02", "1986-03"]
    quarterly = m3_frame_from_mcomp(_mcomp_set(), "Quarterly", horizon=2)
    assert quarterly["ds_source"].tolist() == ["1990-Q1", "1990-Q2", "1990-Q3", "1990-Q4"]

    canonical = _canonicalize(frame, "M3", "Monthly")
    assert canonical["unique_id"].tolist() == ["M3_Monthly_N0002"] * 4 + ["M3_Monthly_N0010"] * 5
    assert canonical["ds_source"].tolist()[:2] == ["1990-01", "1990-02"]
    validate_canonical(canonical, source_dataset="M3_Monthly", frequency="monthly", expected_series=2, expected_min_length=4)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"N0002": _mcomp("N0002", "M1", [1.5, 2.5], [3.5, 4.5], h=3)}, "n=2"),
        ({"N0002": _mcomp("N0002", "M1", [1.5, 2.5], [3.5, 4.5], n=5)}, "n=5"),
        ({"N0002": _mcomp("N0002", "M1", [1.5, np.nan], [3.5, 4.5])}, "non-finite"),
        ({"N0002": _mcomp("N0002", "M3", [1.5, 2.5], [3.5, 4.5])}, "legacy id M3 is not M1"),
        ({"N0002": _mcomp("N0007", "M1", [1.5, 2.5], [3.5, 4.5])}, "field sn is N0007"),
        ({"N0002": {**_mcomp("N0002", "M1", [1.5, 2.5], [3.5, 4.5]),
                    "xx": pd.Series([3.5, 4.5], index=[Fraction(1990) + Fraction(3, 12), Fraction(1990) + Fraction(4, 12)])}},
         "not consecutive"),
    ],
)
def test_mcomp_copy_refuses_inconsistent_series(changes, message):
    with pytest.raises(FrozenDataError, match=message):
        m3_frame_from_mcomp(_mcomp_set(**changes), "Monthly", horizon=2)


def _tsf(tmp_path, rows, horizon=2):
    text = "\n".join(
        ["# M3 monthly – Monash archive", "@relation m3", "@attribute series_name string", "@frequency monthly",
         f"@horizon {horizon}", "@missing false", "@equallength false", "@data"]
        + [f"{name}:1990-01-01 00-00-00:{values}" for name, values in rows]
    )
    path = tmp_path / "m.tsf"
    path.write_text(text + "\n", encoding="cp1252")  # en dash = byte 0x96, as in the real files
    return path


def test_monash_tsf_is_read_at_full_precision(tmp_path):
    header, names, series = read_monash_tsf(_tsf(tmp_path, [("T1", "4812.45,0.1,7"), ("T2", "-1200,3")]))
    assert header["horizon"] == "2" and header["missing"] == "false"
    assert names == ["T1", "T2"]
    assert series[0].tolist() == [4812.45, 0.1, 7.0] and series[0].dtype == np.float64
    with pytest.raises(FrozenDataError, match="missing values"):
        read_monash_tsf(_tsf(tmp_path, [("T1", "1,?,3")]))


def _m3c_sheet(rows, period_col="Starting Month"):
    """A sheet shaped like M3C.xls: Series, N, NF, Category, start, then values and empty cells."""
    width = max(len(values) for *_, values in rows) + 1
    records = [
        [name, len(values), nf, category, year, start] + list(values) + [np.nan] * (width - len(values))
        for name, nf, category, year, start, values in rows
    ]
    columns = ["Series", "N", "NF", "Category", "Starting Year", period_col] + list(range(1, width + 1))
    return pd.DataFrame(records, columns=columns)


MONTHLY_SHEET_ROWS = [
    ("N  12", 2, "MICRO", 1985, 11, [1.5, 2.5, 3.5, 4.5]),
    ("N1402", 2, "OTHER", 0, 0, [5.0, 6.0, 7.0, 8.0, 9.0]),
]


def test_m3c_reader_official_ids_values_and_start_labels():
    frame, notes = m3_frame_from_m3c(_m3c_sheet(MONTHLY_SHEET_ROWS), "Monthly", horizon=2)
    assert frame["unique_id"].tolist() == ["N0012"] * 4 + ["N1402"] * 5
    assert frame["ds"].tolist() == [1, 2, 3, 4, 1, 2, 3, 4, 5]
    assert frame["y"].tolist() == [1.5, 2.5, 3.5, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert frame["ds_source"].tolist() == ["1985-11", "1985-12", "1986-01", "1986-02", "1", "2", "3", "4", "5"]
    assert notes == {"start_period_carried_over": [], "no_start_date_labels_are_positions": ["N1402"]}

    quarterly = _m3c_sheet([("N 646", 2, "MACRO", 1984, 9, [1.0, 2.0, 3.0])], period_col="Starting Quarter")
    frame_q, notes_q = m3_frame_from_m3c(quarterly, "Quarterly", horizon=2)
    assert frame_q["ds_source"].tolist() == ["1986-Q1", "1986-Q2", "1986-Q3"]
    assert notes_q["start_period_carried_over"] == ["N0646"]

    canonical = _canonicalize(frame, "M3", "Monthly")
    assert canonical["unique_id"].tolist() == ["M3_Monthly_N0012"] * 4 + ["M3_Monthly_N1402"] * 5
    validate_canonical(canonical, source_dataset="M3_Monthly", frequency="monthly", expected_series=2, expected_min_length=4)


def _nf_wrong(df):
    df.loc[0, "NF"] = 3
    return df


def _n_too_large(df):
    df.loc[0, "N"] = 5
    return df


def _n_too_small(df):
    df.loc[0, "N"] = 3
    return df


def _gap(df):
    df.loc[0, 2] = np.nan
    return df


def _out_of_order(df):
    return df.iloc[::-1].reset_index(drop=True)


def _duplicated(df):
    df.loc[1, "Series"] = "N0012"
    return df


def _bad_name(df):
    df.loc[0, "Series"] = "X12"
    return df


def _wrong_columns(df):
    return df.rename(columns={"Starting Month": "Starting Quarter"})


@pytest.mark.parametrize(
    "mutate, message",
    [
        (_nf_wrong, "NF=3"),
        (_n_too_large, "N=5"),
        (_n_too_small, "N=3"),
        (_gap, "not a gap-free prefix"),
        (_out_of_order, "official-id order"),
        (_duplicated, "duplicated"),
        (_bad_name, "unexpected M3 series name"),
        (_wrong_columns, "unexpected columns"),
    ],
)
def test_m3c_reader_refuses_inconsistent_rows(mutate, message):
    with pytest.raises(FrozenDataError, match=message):
        m3_frame_from_m3c(mutate(_m3c_sheet(MONTHLY_SHEET_ROWS)), "Monthly", horizon=2)


def test_m3_ids_are_zero_padded_official_ids():
    assert [_m3_id(x) for x in ["N 646", "N1402", "N   1", " N2829 "]] == ["N0646", "N1402", "N0001", "N2829"]
    with pytest.raises(FrozenDataError):
        _m3_id("M1")


def _frame_and_copy():
    frame, _ = m3_frame_from_m3c(_m3c_sheet(MONTHLY_SHEET_ROWS), "Monthly", horizon=2)
    copy = {"N0012": np.array([1.5, 2.5, 3.5, 4.5]), "N1402": np.array([5.0, 6.0, 7.0, 8.0, 9.0])}
    return frame, copy


def test_compare_with_copy_accepts_identical_series_and_reports_counts():
    frame, copy = _frame_and_copy()
    report = compare_with_copy(frame, copy, "Monthly", known={})
    assert report == {"series_compared": 2, "values_compared": 9, "values_identical": 9, "differences": []}


def test_compare_with_copy_requires_documented_differences_exactly():
    frame, copy = _frame_and_copy()
    changed = {**copy, "N1402": np.array([5.0, 6.0, -7.0, 8.0, 9.0])}
    known = {("Monthly", "N1402", 3): (7.0, -7.0)}
    report = compare_with_copy(frame, changed, "Monthly", known=known)
    assert report["differences"] == [{"series": "N1402", "t": 3, "m3c": 7.0, "copy": -7.0}]
    assert report["values_identical"] == 8
    with pytest.raises(FrozenDataError, match="undocumented"):
        compare_with_copy(frame, changed, "Monthly", known={})
    with pytest.raises(FrozenDataError, match="documented differences not found"):
        compare_with_copy(frame, copy, "Monthly", known=known)
    one_ulp = {**copy, "N0012": np.array([1.5, 2.5, 3.5, np.nextafter(4.5, 5)])}
    with pytest.raises(FrozenDataError, match="undocumented"):
        compare_with_copy(frame, one_ulp, "Monthly", known={})


def test_compare_with_copy_refuses_different_ids_or_lengths():
    frame, copy = _frame_and_copy()
    with pytest.raises(FrozenDataError, match="ids differ"):
        compare_with_copy(frame, {"N0012": copy["N0012"]}, "Monthly", known={})
    with pytest.raises(FrozenDataError, match="5 values, copy 4"):
        compare_with_copy(frame, {**copy, "N1402": copy["N1402"][:4]}, "Monthly", known={})


def test_monash_series_are_matched_to_official_ids_by_file_order():
    series = [np.array([1.0]), np.array([2.0])]
    assert monash_by_id(["T1", "T2"], series, ["N0012", "N1402"])["N1402"].tolist() == [2.0]
    with pytest.raises(FrozenDataError, match="2 series"):
        monash_by_id(["T2", "T1"], series, ["N0012", "N1402"])
    with pytest.raises(FrozenDataError, match="expected 3"):
        monash_by_id(["T1", "T2"], series, ["N0012", "N1402", "N1403"])


def test_supplied_and_downloaded_files_must_have_the_pinned_checksum(tmp_path):
    source = tmp_path / "file.bin"
    source.write_bytes(b"not the pinned file")
    dest = tmp_path / "raw" / "file.bin"
    _copy_pinned(source, dest, sha256_file(source))
    assert dest.read_bytes() == b"not the pinned file"
    with pytest.raises(FrozenDataError, match="pinned"):
        _copy_pinned(source, tmp_path / "raw" / "other.bin", M3C_SHA256)
    with pytest.raises(FrozenDataError, match="does not exist"):
        _copy_pinned(tmp_path / "missing.xls", tmp_path / "raw" / "x.xls", M3C_SHA256)

    fetched = tmp_path / "raw" / "archive.tar.gz"
    _fetch(source.as_uri(), fetched, sha256_file(source))
    assert fetched.read_bytes() == b"not the pinned file"
    with pytest.raises(FrozenDataError, match="pinned"):
        _fetch(source.as_uri(), fetched, MCOMP_SHA256)


# --- freeze refuses unsafe invocations before touching the network ------------------------

def test_freeze_refuses_non_empty_directory_bad_commit_and_wrong_m3c(tmp_path):
    fake_m3c = tmp_path / "M3C.xls"
    fake_m3c.write_bytes(b"not the competition file")
    with pytest.raises(FrozenDataError, match="not empty"):
        freeze(tmp_path, "a" * 40, fake_m3c)
    with pytest.raises(FrozenDataError, match="commit"):
        freeze(tmp_path / "new", "not-a-hash", fake_m3c)
    with pytest.raises(FrozenDataError, match="not the pinned M3C.xls"):
        freeze(tmp_path / "new", "a" * 40, fake_m3c)
    with pytest.raises(FrozenDataError, match="not the pinned M3C.xls"):
        freeze(tmp_path / "new", "a" * 40, tmp_path / "absent.xls")
    assert not (tmp_path / "new").exists()
