"""Frozen-data layer: validation rules, content hash, verified loading (protocol D2, test U18).

All data here is synthetic and built inside the tests; the pipeline itself never
generates data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helpers_frozen import make_canonical, spec, write_frozen
from src.data.freeze_data import _canonicalize, freeze, read_m4_official
from src.data.frozen import (
    COLUMNS,
    FrozenDataError,
    content_sha256,
    load_frequency,
    load_manifest,
    load_source,
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


def test_canonicalize_m3_keeps_iso_dates_and_positions():
    y_df = pd.DataFrame({
        "unique_id": ["M2", "M2", "M1", "M1", "M1"],
        "ds": pd.to_datetime(["1990-02-28", "1990-01-31", "1985-03-31", "1985-01-31", "1985-02-28"]),
        "y": [5.0, 4.0, 3.0, 1.0, 2.0],
    })
    canonical = _canonicalize(y_df, "M3", "Monthly")
    assert canonical["unique_id"].tolist() == ["M3_Monthly_M1"] * 3 + ["M3_Monthly_M2"] * 2
    assert canonical["t"].tolist() == [1, 2, 3, 1, 2]
    assert canonical["y"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert canonical["ds_source"].tolist() == ["1985-01-31", "1985-02-28", "1985-03-31", "1990-01-31", "1990-02-28"]
    validate_canonical(canonical, source_dataset="M3_Monthly", frequency="monthly", expected_series=2, expected_min_length=2)


# --- freeze refuses unsafe invocations before touching the network ------------------------

def test_freeze_refuses_non_empty_directory_and_bad_commit(tmp_path):
    (tmp_path / "existing.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FrozenDataError, match="not empty"):
        freeze(tmp_path, "a" * 40)
    with pytest.raises(FrozenDataError, match="commit"):
        freeze(tmp_path / "new", "not-a-hash")
