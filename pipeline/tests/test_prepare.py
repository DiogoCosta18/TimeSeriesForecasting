"""Stage R1 prepare on a small synthetic frozen copy (protocol Sections 3.2-3.6; tests U4; gates G7, G8)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from helpers_frozen import write_frozen
from src.data.frozen import COLUMNS, sha256_file
from src.stages.prepare import PrepareError, run_prepare

COMMIT = "a" * 40
CONFIG = {
    "random_seed": 7,
    "frequencies": {
        "monthly": {"season_length": 12, "horizon": 18, "m3_group": "Monthly", "m4_group": "Monthly"},
        "quarterly": {"season_length": 4, "horizon": 8, "m3_group": "Quarterly", "m4_group": "Quarterly"},
    },
    "validation": {"n_windows": 3},
    "sampling": {"n_per_source": 6, "n_strata": 3},
    "buckets": {"min_share": 0.2},
    "tuning_set": {"n_per_source": 6},
}
SPEC = {"Monthly": (12, 120, 100), "Quarterly": (4, 50, 40)}  # m, regular length, short length


def _canonical(key: str, seed: int) -> pd.DataFrame:
    group = key.split("_")[1]
    m, n, n_short = SPEC[group]
    rng = np.random.default_rng(seed)
    series = {}
    for i in range(8):
        t = np.arange(n)
        series[f"R{i}"] = 100 + 0.2 * t + 6 * np.sin(2 * np.pi * t / m + i) + rng.normal(0, 1.5 + i / 4, n)
    series["SHORT"] = 100 + rng.normal(0, 2, n_short)       # history below L
    flat = 100 + rng.normal(0, 2, n)
    flat[: n - 3 * {12: 18, 4: 8}[m]] = 50.0                  # constant history: features undefined
    series["FLAT"] = flat
    rows = [
        {"unique_id": f"{key}_{sid}", "source_dataset": key, "frequency": group.lower(), "t": t, "y": float(v), "ds_source": str(t)}
        for sid, values in series.items() for t, v in enumerate(values, start=1)
    ]
    return pd.DataFrame(rows)[COLUMNS].astype({"t": "int64", "y": "float64"})


@pytest.fixture(scope="module")
def frozen_copy(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    frames = {key: _canonical(key, i) for i, key in enumerate(["M3_Monthly", "M4_Monthly", "M3_Quarterly", "M4_Quarterly"])}
    _, manifest_path = write_frozen(root, frames)
    return root, manifest_path


@pytest.fixture(scope="module")
def prepared(frozen_copy, tmp_path_factory):
    data_dir, manifest_path = frozen_copy
    run = tmp_path_factory.mktemp("run")
    bundle = run_prepare(CONFIG, run, data_dir, COMMIT, manifest_path=manifest_path)
    tables = {name: pd.read_parquet(run / "prepare" / entry["file"]) for name, entry in bundle["files"].items()}
    return run, bundle, tables


def test_bundle_files_and_counts(prepared):
    run, bundle, tables = prepared
    stored = json.loads((run / "prepare" / "bundle.json").read_text(encoding="utf-8"))
    assert stored["bundle_sha256"] == bundle["bundle_sha256"]
    for entry in bundle["files"].values():
        assert sha256_file(run / "prepare" / entry["file"]) == entry["sha256"]
    for frequency, sources in bundle["counts"].items():
        for counts in sources.values():
            assert (counts["pool"], counts["length_eligible"], counts["eligible"]) == (10, 9, 8)
    assert len(bundle["bucket_summary"]) == 12  # six features x two frequencies
    assert tables["samples"].groupby(["frequency", "feature_name", "source_dataset"]).size().eq(6).all()


def test_u4_only_eligible_series_reach_samples_tuning_and_cutoffs(prepared):
    _, _, tables = prepared
    elig = tables["eligibility"].set_index("unique_id")
    eligible = set(elig.index[elig["eligible"]])
    for name in ["samples", "buckets", "tuning_set", "cutoffs"]:
        assert set(tables[name]["unique_id"]) <= eligible, name
    assert set(tables["cutoffs"]["unique_id"]) == eligible
    assert elig.loc["M3_Monthly_SHORT", "ineligible_reason"] == "history_length 46 < L 54"
    assert elig.loc["M4_Quarterly_SHORT", "ineligible_reason"] == "history_length 16 < L 20"
    assert "feature_non_normality=undefined:constant" in elig.loc["M3_Monthly_FLAT", "ineligible_reason"]


def test_g7_features_and_tuning_validation_end_at_the_first_cutoff(prepared):
    _, _, tables = prepared
    first = tables["cutoffs"].query("window == 0").set_index("unique_id")["train_end_idx"]
    features = tables["features"].query("eligible").set_index("unique_id")["history_end_t"]
    assert (features == first.loc[features.index]).all()
    tuning = tables["tuning_set"].set_index("unique_id")
    assert (tuning["tuning_validation_end_t"] == first.loc[tuning.index]).all()
    block = (tuning["tuning_validation_end_t"] - tuning["tuning_train_end_t"]).groupby(tuning["frequency"]).unique()
    assert block.to_dict() == {"monthly": [18], "quarterly": [8]}  # one validation block of h (D13)


def test_same_inputs_give_the_same_bundle(frozen_copy, prepared, tmp_path):
    data_dir, manifest_path = frozen_copy
    _, bundle, _ = prepared
    again = run_prepare(CONFIG, tmp_path, data_dir, COMMIT, manifest_path=manifest_path)
    assert again["bundle_sha256"] == bundle["bundle_sha256"]
    assert again["files"] == bundle["files"]


def test_a_bundle_is_never_overwritten_or_resumed_with_other_inputs(frozen_copy, prepared, tmp_path):
    data_dir, manifest_path = frozen_copy
    run, _, _ = prepared
    with pytest.raises(PrepareError, match="never overwritten"):
        run_prepare(CONFIG, run, data_dir, COMMIT, manifest_path=manifest_path)
    with pytest.raises(PrepareError, match="40-character"):
        run_prepare(CONFIG, tmp_path, data_dir, "not-a-commit", manifest_path=manifest_path)
    (tmp_path / "prepare").mkdir()
    (tmp_path / "prepare" / "prepare_state.json").write_text(json.dumps({"code_commit": "b" * 40}), encoding="utf-8")
    with pytest.raises(PrepareError, match="other inputs"):
        run_prepare(CONFIG, tmp_path, data_dir, COMMIT, manifest_path=manifest_path)
