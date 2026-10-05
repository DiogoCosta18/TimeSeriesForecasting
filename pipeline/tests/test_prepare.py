"""Stage R1 prepare on a small synthetic frozen copy (protocol Sections 3.2-3.6; tests U4; gates G7, G8)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from helpers_run import COMMIT, CONFIG, synthetic_frozen_copy
from src.data.frozen import sha256_file
from src.stages.prepare import PrepareError, run_prepare

@pytest.fixture(scope="module")
def frozen_copy(tmp_path_factory):
    return synthetic_frozen_copy(tmp_path_factory.mktemp("data"))


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


def test_a_feature_subset_samples_only_those_features_with_the_full_runs_seeds(frozen_copy, prepared, tmp_path):
    """The pilot's subset (Section 8.1): same eligibility; each feature drawn as in the full run."""
    data_dir, manifest_path = frozen_copy
    _, _, full = prepared
    subset = {**CONFIG, "features": ["feature_evolving_seasonality", "feature_nonlinearity"]}
    bundle = run_prepare(subset, tmp_path, data_dir, COMMIT, manifest_path=manifest_path)
    samples = pd.read_parquet(tmp_path / "prepare" / "samples.parquet")
    assert set(samples["feature_name"]) == set(subset["features"]) and len(bundle["bucket_summary"]) == 4
    expected = full["samples"][full["samples"]["feature_name"].isin(subset["features"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(samples, expected)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "prepare" / "eligibility.parquet"), full["eligibility"])
