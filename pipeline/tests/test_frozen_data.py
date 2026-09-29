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

from src.data.freeze_data import check_m4_against_raw, freeze
from src.data.frozen import (
    COLUMNS,
    MANIFEST_FORMAT,
    FrozenDataError,
    content_sha256,
    load_frequency,
    load_manifest,
    load_source,
    sha256_file,
    validate_canonical,
    verify_all,
)


def make_canonical(source_dataset: str = "M3_Monthly", lengths=(5, 7, 6), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frequency = source_dataset.split("_")[1].lower()
    rows = []
    for i, n in enumerate(lengths, start=1):
        for t in range(1, n + 1):
            rows.append({
                "unique_id": f"{source_dataset}_S{i}",
                "source_dataset": source_dataset,
                "frequency": frequency,
                "t": t,
                "y": float(rng.uniform(10, 100)),
                "ds_source": str(t),
            })
    df = pd.DataFrame(rows)[COLUMNS]
    return df.astype({"t": "int64", "y": "float64"})


def spec(df: pd.DataFrame) -> dict:
    lengths = df.groupby("unique_id")["t"].count()
    return {
        "source_dataset": df["source_dataset"].iloc[0],
        "frequency": df["frequency"].iloc[0],
        "expected_series": int(lengths.size),
        "expected_min_length": int(lengths.min()),
    }


def write_frozen(tmp_path: Path, frames: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    frozen_dir = tmp_path / "frozen"
    frozen_dir.mkdir()
    entries = {}
    for key, df in frames.items():
        path = frozen_dir / f"{key}.parquet"
        df.to_parquet(path, engine="pyarrow", index=False, compression="zstd")
        stats = validate_canonical(df, **spec(df))
        entries[key] = {
            "file": path.name,
            "frequency": df["frequency"].iloc[0],
            "horizon": 2,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "content_sha256": content_sha256(df),
            **stats,
        }
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(json.dumps({"format": MANIFEST_FORMAT, "frozen_files": entries}), encoding="utf-8")
    return frozen_dir, manifest_path


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


# --- M4 cross-check against the official files -----------------------------------------

def _m4_raw(tmp_path: Path, train_rows, test_rows) -> Path:
    base = tmp_path / "raw" / "m4" / "datasets"
    base.mkdir(parents=True)
    width = max(len(r) for _, r in train_rows)
    train = pd.DataFrame([[sid] + list(r) + [np.nan] * (width - len(r)) for sid, r in train_rows],
                         columns=["V1"] + [f"V{i}" for i in range(2, width + 2)])
    test = pd.DataFrame([[sid] + list(r) for sid, r in test_rows],
                        columns=["V1"] + [f"V{i}" for i in range(2, len(test_rows[0][1]) + 2)])
    train.to_csv(base / "Monthly-train.csv", index=False)
    test.to_csv(base / "Monthly-test.csv", index=False)
    return tmp_path / "raw"


def _m4_frozen(series: dict[str, list[float]]) -> pd.DataFrame:
    rows = [
        {"unique_id": f"M4_Monthly_{sid}", "source_dataset": "M4_Monthly", "frequency": "monthly",
         "t": t, "y": float(v), "ds_source": str(t)}
        for sid, values in series.items() for t, v in enumerate(values, start=1)
    ]
    return pd.DataFrame(rows)[COLUMNS]


def test_m4_check_accepts_exact_train_plus_test(tmp_path):
    raw = _m4_raw(tmp_path, [("M1", [1.0, 2.0, 3.0]), ("M2", [4.0, 5.0])], [("M1", [9.0, 8.0]), ("M2", [7.0, 6.0])])
    frozen = _m4_frozen({"M1": [1, 2, 3, 9, 8], "M2": [4, 5, 7, 6]})
    check_m4_against_raw(frozen, raw, "Monthly", horizon=2)


def test_m4_check_rejects_gap_in_training_row(tmp_path):
    raw = _m4_raw(tmp_path, [("M1", [1.0, np.nan, 3.0])], [("M1", [9.0, 8.0])])
    frozen = _m4_frozen({"M1": [1, 3, 9, 8]})
    with pytest.raises(FrozenDataError, match="gap"):
        check_m4_against_raw(frozen, raw, "Monthly", horizon=2)


def test_m4_check_rejects_values_that_differ_from_the_official_files(tmp_path):
    raw = _m4_raw(tmp_path, [("M1", [1.0, 2.0])], [("M1", [9.0, 8.0])])
    frozen = _m4_frozen({"M1": [1, 2, 9, 8.0000001]})
    with pytest.raises(FrozenDataError, match="differs"):
        check_m4_against_raw(frozen, raw, "Monthly", horizon=2)


def test_m4_check_rejects_wrong_test_length(tmp_path):
    raw = _m4_raw(tmp_path, [("M1", [1.0, 2.0])], [("M1", [9.0, 8.0, 7.0])])
    frozen = _m4_frozen({"M1": [1, 2, 9, 8, 7]})
    with pytest.raises(FrozenDataError, match="exactly 2"):
        check_m4_against_raw(frozen, raw, "Monthly", horizon=2)


# --- freeze refuses unsafe invocations before touching the network ------------------------

def test_freeze_refuses_non_empty_directory_and_bad_commit(tmp_path):
    (tmp_path / "existing.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FrozenDataError, match="not empty"):
        freeze(tmp_path, "a" * 40)
    with pytest.raises(FrozenDataError, match="commit"):
        freeze(tmp_path / "new", "not-a-hash")
