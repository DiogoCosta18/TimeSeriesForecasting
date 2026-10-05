"""Stage R4 merge and gates G1-G13 on a complete synthetic run (tests U14, U15; Section 7.4).

The run is produced by the real stages and engine with cheap stand-in fits
(helpers_run.fake_fits); one feature (as in the pilot) and one extra seed keep it small.
"""
from __future__ import annotations

import copy
import shutil

import numpy as np
import pandas as pd
import pytest

from helpers_run import COMMIT, CONFIG, evaluated_run
from src.stages.io import read_json, safe_name, write_json, write_parquet
from src.stages.merge import VECTORS, MergeError, check_twins, run_merge
from src.validation import gates
from src.validation.gates import run_gates

SMALL = {**copy.deepcopy(CONFIG), "features": ["feature_evolving_seasonality"], "seed_check_seeds": [8]}
FIX = "b" * 40


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    return evaluated_run(tmp_path_factory.mktemp("evaluated"), SMALL)


@pytest.fixture
def run(evaluated, tmp_path, monkeypatch):
    source, _, manifest_path = evaluated
    shutil.copytree(source, tmp_path / "run")
    monkeypatch.setenv("RERUN_CODE_COMMIT", COMMIT)
    return tmp_path / "run", manifest_path


def _record_path(run_dir, task_id):
    (path,) = run_dir.glob(f"evaluate/*/tasks/{safe_name(task_id)}.done.json")
    return path


def _rewrite(run_dir, task_id, rows_fn=None, record_fn=None):
    """Change a task's output and keep its completion record consistent with it."""
    record_path = _record_path(run_dir, task_id)
    record = read_json(record_path)
    output = record_path.with_name(record["file"])
    rows = pd.read_parquet(output)
    if rows_fn is not None:
        rows = rows_fn(rows)
    record["sha256"], record["rows"] = write_parquet(output, rows), len(rows)
    if record_fn is not None:
        record_fn(record)
    write_json(record_path, record)


def _task_ids(run_dir, scope=None, family=None):
    ids = [read_json(p)["task_id"] for p in sorted(run_dir.glob("evaluate/*/tasks/*.done.json"))]
    if scope is not None:
        ids = [t for t in ids if t.startswith("eval|") and t.split("|")[5] == scope]
    if family == "statistical":
        ids = [t for t in ids if t.startswith("stat|")]
    return ids


def test_complete_run_merges_reproducibly_and_passes_every_gate(run):
    run_dir, manifest_path = run
    first = run_merge(run_dir, manifest_path)
    assert run_merge(run_dir, manifest_path)["result_sha256"] == first["result_sha256"]
    assert first["completed"] == first["expected"] and first["missing"] == first["listed_failures"] == 0
    record = read_json(run_dir / "merged" / "merge.json")
    counts = record["grid"]["counts"]
    assert (counts["cohort_global"], counts["seed_check"], counts["cohort_statistical_views"]) == (54, 42, 18)
    rows = pd.read_parquet(run_dir / "merged" / "rows.parquet")
    assert not set(VECTORS) & set(rows.columns) and len(rows) == record["rows"]["trained"]
    assert rows.groupby("task_id").size().to_dict() == {t: e["rows"] for t, e in record["tasks"].items()}

    report = run_gates(run_dir, manifest_path)
    assert report["passed"], {k: v for k, v in report["gates"].items() if not v["passed"]}
    assert report["gates"]["G10"]["median_stl_ac_over_stl_sn"] == {"ml": 3.0, "neural": 3.0, "transformer": 3.0}
    assert report["merge_result_sha256"] == record["result_sha256"]
    with pytest.raises(MergeError, match="frozen"):
        run_merge(run_dir, manifest_path)


def test_incomplete_grid_is_merged_and_reported_by_g5_and_g6(run):
    run_dir, manifest_path = run
    high = _task_ids(run_dir, scope="high")
    stl_ac = next(t for t in high if "|stl_ac|" in t)
    _record_path(run_dir, stl_ac).unlink()
    assert run_merge(run_dir, manifest_path)["missing"] == 1
    report = run_gates(run_dir, manifest_path)
    assert not report["passed"]
    assert [k for k, v in report["gates"].items() if not v["passed"]] == ["G5"]
    assert report["gates"]["G5"]["missing"] == [stl_ac]

    direct = next(t for t in high if "|direct|" in t)  # its STL rows lose their Direct twin
    _record_path(run_dir, direct).unlink()
    run_merge(run_dir, manifest_path)
    report = run_gates(run_dir, manifest_path)
    assert [k for k, v in report["gates"].items() if not v["passed"]] == ["G5", "G6"]
    assert report["gates"]["G6"]["stl_without_direct_twin"] > 0


@pytest.mark.parametrize("field", ["environment_lock_sha256", "data_manifest_sha256", "bundle_sha256", "configs_sha256"])
def test_u15_merge_refuses_any_other_provenance(run, field):
    run_dir, manifest_path = run
    task_id = _task_ids(run_dir, scope="cohort")[0]

    def other(record):
        record["provenance"][field] = "0" * 64

    _rewrite(run_dir, task_id, rows_fn=lambda r: r.assign(**{field: "0" * 64}), record_fn=other)
    with pytest.raises(MergeError, match=f"{field} .* differs from the run's"):
        run_merge(run_dir, manifest_path)


def test_u15_rows_must_carry_their_records_provenance_and_outputs_must_be_unchanged(run):
    run_dir, manifest_path = run
    run_dir2 = run_dir.parent / "run2"
    shutil.copytree(run_dir, run_dir2)
    task_id = _task_ids(run_dir, scope="cohort")[0]
    _rewrite(run_dir, task_id, rows_fn=lambda r: r.assign(configs_sha256="0" * 64))
    with pytest.raises(MergeError, match="rows carry other configs_sha256"):
        run_merge(run_dir, manifest_path)

    other = _task_ids(run_dir2, family="statistical")[0]
    output = _record_path(run_dir2, other).with_name(read_json(_record_path(run_dir2, other))["file"])
    pd.read_parquet(output).iloc[:-1].to_parquet(output, index=False)
    with pytest.raises(MergeError, match="output missing or changed"):
        run_merge(run_dir2, manifest_path)


def test_another_commit_is_accepted_only_as_a_registered_d16_rerun(run):
    run_dir, manifest_path = run
    task_id = _task_ids(run_dir, scope="low")[0]

    def fixed(record):
        record["provenance"]["code_commit"] = FIX

    _rewrite(run_dir, task_id, rows_fn=lambda r: r.assign(code_commit=FIX), record_fn=fixed)
    with pytest.raises(MergeError, match="not a registered D16 rerun"):
        run_merge(run_dir, manifest_path)
    shard_dir = _record_path(run_dir, task_id).parents[1]
    write_json(shard_dir / "d16_reruns.json", {task_id: {"main_commit": COMMIT, "fix_commit": FIX, "failure": {},
                                                         "rerun_at_utc": "2026-10-05T00:00:00Z"}})
    run_merge(run_dir, manifest_path)
    g4 = run_gates(run_dir, manifest_path)["gates"]["G4"]
    assert g4["passed"] and g4["d16_reruns"] == {task_id: FIX}


def test_merge_refuses_rows_that_disagree_with_their_task(run):
    run_dir, manifest_path = run
    task_id = _task_ids(run_dir, scope="cohort")[0]

    def wrong_forecast(rows):
        rows = rows.copy()
        rows.at[0, "yhat"] = list(np.asarray(rows.at[0, "yhat"]) + 1.0)
        return rows

    _rewrite(run_dir, task_id, rows_fn=wrong_forecast)
    with pytest.raises(MergeError, match="does not match its hash"):
        run_merge(run_dir, manifest_path)
    _rewrite(run_dir, task_id, rows_fn=lambda r: r.iloc[1:])
    with pytest.raises(MergeError, match="series and windows differ"):
        run_merge(run_dir, manifest_path)


def test_merge_refuses_a_study_that_no_longer_matches_the_frozen_configuration(run):
    run_dir, manifest_path = run
    (record_path,) = run_dir.glob("tune/ml-monthly/tasks/tune__Ridge__monthly__raw.done.json")
    record = read_json(record_path)
    output = record_path.with_name(record["file"])
    entry = read_json(output)
    write_json(output, {**entry, "params": {"alpha": 2.0}})
    record["sha256"] = gates.sha256_file(output)
    write_json(record_path, record)
    with pytest.raises(MergeError, match="does not match its frozen configuration"):
        run_merge(run_dir, manifest_path)


@pytest.fixture
def merged(run):
    run_dir, manifest_path = run
    run_merge(run_dir, manifest_path)
    record, rows, failed = gates.load_merged(run_dir)
    tables = {n: pd.read_parquet(run_dir / "prepare" / f"{n}.parquet")
              for n in ["eligibility", "features", "samples", "buckets", "bucket_summary", "tuning_set", "cutoffs"]}
    return record, rows, failed, tables, read_json(run_dir / "configs_frozen.json")


def test_u14_every_tercile_row_has_exactly_one_cohort_twin(merged):
    _, rows, _, _, _ = merged
    assert check_twins(rows)["passed"]
    cohort = rows.index[(rows["scope"] == "cohort") & (rows["family"] == "ml") & (rows["seed"] == SMALL["random_seed"])]
    assert check_twins(rows.drop(index=cohort[:1]))["missing_twin"] == 1
    assert check_twins(pd.concat([rows, rows.loc[cohort[:1]]]))["duplicated_cohort_rows"] == 1
    tercile = rows.index[rows["scope"] == "low"][0]
    moved = rows.copy()
    moved.loc[tercile, "bucket"] = "High"
    result = check_twins(moved)
    assert not result["passed"] and result["bucket_not_scope"] == 1 and result["twin_in_other_bucket"] == 1


def test_each_gate_fails_on_its_defect(merged):
    record, rows, failed, tables, frozen = merged
    assert gates.g1_trained_rows_only(rows, failed, record)["passed"]
    assert not gates.g1_trained_rows_only(rows.assign(components='[{"target": "raw"}]'), failed, record)["passed"]

    assert gates.g6_pairing(rows, failed)["passed"]
    direct = rows.index[rows["strategy"] == "direct"][0]
    assert gates.g6_pairing(rows.drop(index=direct), failed)["stl_without_direct_twin"] == 2

    bad = rows.copy()
    bad.loc[bad.index[0], "mase"] = np.nan
    assert gates.g9_finite_metrics(bad)["non_finite_values"] == 1 and not gates.g9_finite_metrics(bad)["passed"]
    capped = rows.copy()
    capped.loc[capped.index[: int(0.03 * len(capped)) + 1], "relnaive"] = 50.0
    assert not gates.g9_finite_metrics(capped)["passed"]

    slow = rows.copy()
    slow.loc[(slow["strategy"] == "stl_ac") & (slow["family"] == "neural"), "fit_seconds"] *= 2
    g10 = gates.g10_time_ratio(slow)
    assert not g10["passed"] and g10["median_stl_ac_over_stl_sn"]["neural"] == 6.0

    late = {**tables, "features": tables["features"].assign(history_end_t=tables["features"]["history_end_t"] + 1)}
    assert not gates.g7_no_leakage(late, frozen)["passed"]
    unchecked = copy.deepcopy(frozen)
    next(iter(unchecked["entries"].values()))["g7_tuning_end_equals_first_cutoff"] = False
    assert not gates.g7_no_leakage(tables, unchecked)["passed"]

    elig = tables["eligibility"].copy()
    elig.loc[elig.index[elig["eligible"]][0], "history_length"] = 3
    assert gates.g8_minimum_history({**tables, "eligibility": elig}, SMALL)["eligible_below_L"] == 1

    shifted = tables["samples"].copy()
    m4 = shifted["source_dataset"] == "M4_Monthly"
    shifted.loc[m4, "feature_value"] += 1000.0
    g11 = gates.g11_sampling({**tables, "samples": shifted})
    assert not g11["passed"] and g11["failing"] == ["feature_evolving_seasonality|monthly|M4_Monthly"]

    flipped = tables["bucket_summary"].copy()
    flipped.loc[flipped.index[0], "degenerate"] = not flipped.loc[flipped.index[0], "degenerate"]
    assert not gates.g12_buckets({**tables, "bucket_summary": flipped}, SMALL)["passed"]
