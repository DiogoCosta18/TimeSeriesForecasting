"""Stage R5 analyse on a gated synthetic run (protocol Section 5, Table 6; I3 on synthetic data).

The run comes from the real stages and engine with stand-in fits (helpers_run); I3
proper runs on the pilot outputs.
"""
from __future__ import annotations

import copy
import shutil

import pytest

from helpers_run import COMMIT, CONFIG, evaluated_run
from src.analysis.data import AnalysisError
from src.analysis.run import OUTPUTS, run_analysis
from src.data.frozen import sha256_file
from src.stages.io import read_json, write_json
from src.stages.merge import run_merge
from src.validation.gates import run_gates

SMALL = {**copy.deepcopy(CONFIG), "features": ["feature_evolving_seasonality"], "seed_check_seeds": [8]}


@pytest.fixture(scope="module")
def gated(tmp_path_factory):
    run, _, manifest_path = evaluated_run(tmp_path_factory.mktemp("analysis"), SMALL)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RERUN_CODE_COMMIT", COMMIT)
        run_merge(run, manifest_path)
        assert run_gates(run, manifest_path)["passed"]
    return run, manifest_path


@pytest.fixture
def run(gated, tmp_path, monkeypatch):
    source, manifest_path = gated
    shutil.copytree(source, tmp_path / "run")
    monkeypatch.setenv("RERUN_CODE_COMMIT", COMMIT)
    return tmp_path / "run", manifest_path


def test_analysis_refuses_ungated_failed_or_stale_gate_reports(run, tmp_path):
    run_dir, manifest_path = run
    gates_path = run_dir / "merged" / "gates.json"
    report = read_json(gates_path)
    write_json(gates_path, {**report, "passed": False, "gates": {**report["gates"], "G5": {"passed": False}}})
    with pytest.raises(AnalysisError, match=r"gates failed: \['G5'\]"):
        run_analysis(run_dir, tmp_path / "a", manifest_path)
    write_json(gates_path, {**report, "merge_result_sha256": "0" * 64})
    with pytest.raises(AnalysisError, match="another merge"):
        run_analysis(run_dir, tmp_path / "b", manifest_path)
    gates_path.unlink()
    with pytest.raises(AnalysisError, match="have not been run"):
        run_analysis(run_dir, tmp_path / "c", manifest_path)


def test_every_output_of_table_6_is_produced_reproducibly(run, tmp_path):
    run_dir, manifest_path = run
    first = run_analysis(run_dir, tmp_path / "first", manifest_path)
    record = read_json(tmp_path / "first" / "analysis.json")
    expected = {p for paths in OUTPUTS.values() for p in paths}
    assert expected <= set(record["files"]) and first["files"] == len(record["files"])
    for path, digest in record["files"].items():
        assert sha256_file(tmp_path / "first" / path) == digest
    assert record["merge_result_sha256"] == read_json(run_dir / "merged" / "merge.json")["result_sha256"]

    run_analysis(run_dir, tmp_path / "second", manifest_path)
    again = read_json(tmp_path / "second" / "analysis.json")["files"]
    assert again == record["files"]                     # tables and figures byte for byte
    with pytest.raises(AnalysisError, match="never overwritten"):
        run_analysis(run_dir, tmp_path / "first", manifest_path)
