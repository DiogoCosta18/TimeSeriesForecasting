"""The experiment's data entry point reads only the verified frozen copy (protocol D2, U18).

Also guards against the defect of the earlier runs (F4): the old loader swallowed
every exception and substituted synthetic series without saying so.
"""
from __future__ import annotations

import ast
import inspect

import pandas as pd
import pytest

import src.data.load_m_datasets as loader
from helpers_frozen import make_canonical, write_frozen
from src.data.frozen import FrozenDataError, load_manifest, sha256_file
from src.data.load_m_datasets import (
    CALENDAR_FREQ,
    CALENDAR_ORIGIN,
    DATA_DIR_ENV,
    OUTPUT_COLUMNS,
    frozen_data_provenance,
    load_dataset_pair,
    resolve_data_dir,
)

MONTHLY = {"frequency": "monthly", "m3_group": "Monthly", "m4_group": "Monthly"}


@pytest.fixture
def frozen_copy(tmp_path):
    frames = {
        "M3_Monthly": make_canonical("M3_Monthly", lengths=(5, 7), seed=1),
        "M4_Monthly": make_canonical("M4_Monthly", lengths=(9, 6, 8), seed=2),
        "M3_Quarterly": make_canonical("M3_Quarterly", lengths=(4, 5), seed=3),
        "M4_Quarterly": make_canonical("M4_Quarterly", lengths=(6,), seed=4),
    }
    _, manifest_path = write_frozen(tmp_path, frames)
    return tmp_path, manifest_path, frames


def test_loads_m3_and_m4_of_one_frequency_on_one_regular_calendar(frozen_copy):
    data_dir, manifest_path, frames = frozen_copy
    data = load_dataset_pair(MONTHLY, data_dir, load_manifest(manifest_path))

    assert list(data.columns) == OUTPUT_COLUMNS
    assert set(data["source_dataset"]) == {"M3_Monthly", "M4_Monthly"}
    expected = pd.concat([frames["M3_Monthly"], frames["M4_Monthly"]]).sort_values(["unique_id", "t"], kind="mergesort")
    assert data["unique_id"].tolist() == expected["unique_id"].tolist()
    assert data["y"].tolist() == expected["y"].tolist()
    for _, g in data.groupby("unique_id"):
        calendar = pd.date_range(CALENDAR_ORIGIN, periods=len(g), freq=CALENDAR_FREQ["monthly"])
        assert g["t"].tolist() == list(range(1, len(g) + 1))
        assert (g["ds"].to_numpy() == calendar.to_numpy()).all()
        assert pd.infer_freq(g["ds"]) == "ME"


def test_quarterly_uses_the_quarterly_calendar(frozen_copy):
    data_dir, manifest_path, _ = frozen_copy
    cfg = {"frequency": "quarterly", "m3_group": "Quarterly", "m4_group": "Quarterly"}
    data = load_dataset_pair(cfg, data_dir, load_manifest(manifest_path))
    assert set(data["source_dataset"]) == {"M3_Quarterly", "M4_Quarterly"}
    assert pd.infer_freq(data[data["unique_id"] == "M4_Quarterly_S1"]["ds"]).startswith("QE")


def test_calendar_origin_leaves_room_for_the_longest_m4_series():
    # M4 Monthly: at most 2,794 training values + 18 test values, plus forecasts beyond.
    calendar = pd.date_range(CALENDAR_ORIGIN, periods=2794 + 18 + 1000, freq="ME")
    assert calendar[-1].year < 2262


def test_data_location_must_be_given(monkeypatch, frozen_copy):
    data_dir, _, _ = frozen_copy
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    with pytest.raises(FrozenDataError, match="--data-dir"):
        resolve_data_dir(None)
    monkeypatch.setenv(DATA_DIR_ENV, str(data_dir))
    assert resolve_data_dir(None) == data_dir


def test_folder_without_frozen_copy_is_refused(tmp_path):
    with pytest.raises(FrozenDataError, match="no frozen/ folder"):
        resolve_data_dir(tmp_path)


def test_configuration_must_match_the_manifest(frozen_copy):
    data_dir, manifest_path, _ = frozen_copy
    wrong = {**MONTHLY, "m4_group": "Quarterly"}
    with pytest.raises(FrozenDataError, match="configuration expects"):
        load_dataset_pair(wrong, data_dir, load_manifest(manifest_path))


def test_an_altered_file_stops_the_run(frozen_copy):
    data_dir, manifest_path, _ = frozen_copy
    path = data_dir / "frozen" / "M4_Monthly.parquet"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(FrozenDataError, match="checksum mismatch"):
        load_dataset_pair(MONTHLY, data_dir, load_manifest(manifest_path))


def test_provenance_identifies_the_manifest_and_every_file(frozen_copy):
    _, manifest_path, frames = frozen_copy
    record = frozen_data_provenance(manifest_path)
    assert record["manifest_sha256"] == sha256_file(manifest_path)
    assert set(record["frozen_files"]) == set(frames)
    assert all(len(v["sha256"]) == 64 and len(v["content_sha256"]) == 64 for v in record["frozen_files"].values())


def test_loader_has_no_fallback_path():
    """No exception handling, no downloader, no random generator: failure can only stop the run."""
    tree = ast.parse(inspect.getsource(loader))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not {m for m in imported if m.startswith(("datasetsforecast", "numpy", "random"))}
