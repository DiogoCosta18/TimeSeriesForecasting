"""Sensitivity analysis S3, season length 1 for the statistical models (protocol change log v2.12 (e)).

ARIMA and ETS under STL-SN and STL-AC with the STL of D9, fitted with season length 1 on the
targets without seasonality (T + R; T and R) and with season length m on the seasonal component
(variant ``p1``); every series of the union of the feature samples, the three windows of the run,
against the run's Direct forecasts and against the run's own STL forecasts. Everything lives in
RUN/sensitivity_s3/; the stages are those of src.stages.sensitivity:

    freeze    no tuning: the provenance of the freezing stage (the evaluation must share it)
    evaluate  --group cpu (statistical chunks; --workers N)
    analyse   the checks (as gates G1, G4, G5, G6, G9 for S3), the two controls, then the tests:
              per model and strategy against Direct and against the run's STL (two families of four,
              Holm), and per strategy for the family of ARIMA and ETS under p1 and under the run's
              season length (the specification of v1.0 without SARIMA), Holm over four

Controls: ARIMA and ETS under STL-AC with the run's STL and season length on the first chunk of
each frequency must be bit-identical to the run's rows; with ``--reference-s2 RUN2`` (a run whose
S2 ran), the STL-SN rows must be bit-identical to S2's ``stlf`` rows of the same model, series and
window (S2's statistical baseline being identical to this run's, v2.9 (b)).

    python -m src.stages.sensitivity3 STAGE --config C --run RUN [--data-dir D] [--workers N] [--reference-s2 RUN2]

S3 adds no confirmatory hypothesis. Under decision (1) of v2.12 its rows are the main
specification of the statistical STL results (src.analysis.run --statistical-stl).
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.data import ROW_COLUMNS, AnalysisData, AnalysisError, build_instances, per_series, strategy_contrast
from src.analysis.hypotheses import one_sample
from src.data.frozen import DEFAULT_MANIFEST
from src.stages import sensitivity as s1
from src.stages.io import StageError, code_commit, load_bundle, read_json, sha256_json, utc_now, write_json
from src.stages.tasks import FREQUENCIES, STATISTICAL_CHUNK
from src.validation.gates import load_merged

log = logging.getLogger(__name__)

ROOT = "sensitivity_s3"
MODELS = ["ARIMA", "ETS"]
STRATEGIES = ["stl_sn", "stl_ac"]
VARIANT = "p1"
FROZEN_FORMAT = "rerun-s3-configs/1"
CONTROL_PREFIX = "s3ctrl"


def tune_tasks() -> list[dict]:
    return []


def eval_tasks(bucket_summary: pd.DataFrame, samples: pd.DataFrame) -> list[dict]:
    tasks = []
    for frequency in FREQUENCIES:
        n_chunks = math.ceil(samples.loc[samples["frequency"] == frequency, "unique_id"].nunique() / STATISTICAL_CHUNK)
        for model in MODELS:
            for strategy in STRATEGIES:
                for chunk in range(n_chunks):
                    tasks.append({"task_id": f"s3stat|{model}|{frequency}|{strategy}|{VARIANT}|chunk{chunk:03d}",
                                  "kind": "statistical", "group": "cpu", "frequency": frequency, "chunk": chunk,
                                  "model": model, "strategy": strategy, "variant": VARIANT})
            tasks.append({"task_id": f"{CONTROL_PREFIX}|{model}|{frequency}|stl_ac|default|chunk000", "kind": "statistical",
                          "group": "cpu", "frequency": frequency, "chunk": 0, "model": model, "strategy": "stl_ac",
                          "variant": "default"})
    return tasks


S3 = s1.Study("s3", ROOT, tune_tasks, eval_tasks, FROZEN_FORMAT)


# --- analyse ---------------------------------------------------------------------------------------------

KEYS = ["model", "strategy", "unique_id", "window"]


def control_check(s3_rows: pd.DataFrame, run_rows: pd.DataFrame) -> dict:
    """The control rows (the run's STL and season length) against the run's rows: forecast hashes."""
    ctrl = s3_rows[s3_rows["task_id"].astype(str).str.startswith(CONTROL_PREFIX)]
    run = run_rows[run_rows["strategy"] == "stl_ac"][KEYS + ["forecast_hash"]]
    j = ctrl[KEYS + ["forecast_hash"]].merge(run, on=KEYS, how="left", suffixes=("", "_run"), validate="one_to_one")
    return {"rows": int(len(j)), "without_run_row": int(j["forecast_hash_run"].isna().sum()),
            "identical": int((j["forecast_hash"] == j["forecast_hash_run"]).sum()),
            "passed": len(j) > 0 and bool((j["forecast_hash"] == j["forecast_hash_run"]).all())}


def s2_check(s3_rows: pd.DataFrame, reference: Path | None) -> dict:
    """The STL-SN rows against S2's stlf rows of another run with identical statistical forecasts."""
    if reference is None:
        return {"reference": None, "passed": True, "note": "no reference given; checked separately if at all"}
    from src.stages import sensitivity2 as s2
    prepare = Path(reference) / "prepare"
    samples = pd.read_parquet(prepare / "samples.parquet")
    tasks = [t for t in s2.eval_tasks(pd.read_parquet(prepare / "bucket_summary.parquet"), samples)
             if t["kind"] == "statistical" and t["variant"] == "stlf"]
    ref, facts = s1._s1_rows(reference, tasks, s2.ROOT, s1.ANALYSIS_COLUMNS)
    if facts["missing"] or facts["changed"]:
        return {"reference": str(reference), "passed": False, "missing": facts["missing"], "changed": facts["changed"]}
    mine = s3_rows[(s3_rows["strategy"] == "stl_sn") & ~s3_rows["task_id"].astype(str).str.startswith(CONTROL_PREFIX)]
    j = mine[KEYS + ["forecast_hash"]].merge(ref[KEYS + ["forecast_hash"]], on=KEYS, how="outer", suffixes=("", "_s2"),
                                              validate="one_to_one", indicator=True)
    same = (j["_merge"] == "both") & (j["forecast_hash"] == j["forecast_hash_s2"])
    return {"reference": str(reference), "rows": int(len(j)), "identical": int(same.sum()),
            "only_s3": int((j["_merge"] == "left_only").sum()), "only_s2": int((j["_merge"] == "right_only").sum()),
            "passed": bool(same.all())}


def run_analyse(run_dir: Path, config: dict, manifest_path: Path = DEFAULT_MANIFEST, reference_s2: Path | None = None) -> dict:
    run_dir = Path(run_dir)
    out = run_dir / ROOT / "analysis"
    if out.exists() and any(out.iterdir()):
        raise StageError(f"{out} is not empty; S3 outputs are never overwritten")
    commit = code_commit()
    bundle = load_bundle(run_dir, config, manifest_path)
    frozen = s1.load_frozen(run_dir, S3)
    prepare = run_dir / "prepare"
    samples, buckets = pd.read_parquet(prepare / "samples.parquet"), pd.read_parquet(prepare / "buckets.parquet")
    tasks = eval_tasks(pd.read_parquet(prepare / "bucket_summary.parquet"), samples)
    s3_rows, facts = s1._s1_rows(run_dir, tasks, ROOT, s1.ANALYSIS_COLUMNS)
    record, rows, failed = load_merged(run_dir, ROW_COLUMNS + ["forecast_hash"])
    stat = rows[(rows["family"] == "statistical")]

    checks = {"G5_complete": {"expected": len(tasks), "missing": facts["missing"], "changed": facts["changed"]},
              "G4_commit_and_pins": {"commits": facts["commits"], "pinned": facts["pins"], "configs": facts["configs"],
                                     "frozen_configs": frozen["configs_sha256"],
                                     "bundle_matches_run": bundle["bundle_sha256"] == record["provenance"]["bundle_sha256"]}}
    checks["G5_complete"]["passed"] = not facts["missing"] and not facts["changed"]
    checks["G4_commit_and_pins"]["passed"] = (len(facts["commits"]) == 1 and facts["pins"] == [True]
                                              and facts["configs"] == [frozen["configs_sha256"]]
                                              and checks["G4_commit_and_pins"]["bundle_matches_run"])
    trained = s3_rows[s3_rows["status"] == "trained"] if len(s3_rows) else s3_rows
    metrics = ["relnaive_capped", "relnaive", "mase", "smape", "mae", "pocid"]
    nonfinite = int((~np.isfinite(trained[metrics].to_numpy(dtype=float))).any(axis=1).sum()) if len(trained) else 0
    checks["G1_trained_rows"] = {"rows": int(len(s3_rows)), "failed_rows": int(len(s3_rows) - len(trained)),
                                 "listed_task_failures": facts["listed_failures"], "passed": not facts["listed_failures"]}
    checks["G9_finite"] = {"non_finite_rows": nonfinite, "passed": nonfinite == 0}
    if len(s3_rows):
        checks["control_reproduces_run"] = control_check(s3_rows, stat)
        checks["stl_sn_matches_s2_stlf"] = s2_check(s3_rows, reference_s2)

    main_seed = int(config["random_seed"])
    p1 = s3_rows[~s3_rows["task_id"].astype(str).str.startswith(CONTROL_PREFIX)] if len(s3_rows) else s3_rows
    p1_inst_rows = p1.assign(strategy=p1["strategy"].astype(str) + "_p1") if len(p1) else p1
    run_rows = stat[stat["model"].isin(MODELS)]
    both = pd.concat([run_rows[ROW_COLUMNS], p1_inst_rows[ROW_COLUMNS]], ignore_index=True)
    inst = build_instances(both, failed.iloc[0:0], buckets, main_seed)
    inst = inst[inst["main_seed"]]
    vs_direct = pd.concat([strategy_contrast(inst, "direct", f"{s}_p1").assign(strategy_p1=s) for s in STRATEGIES],
                          ignore_index=True)
    vs_run = pd.concat([strategy_contrast(inst, s, f"{s}_p1").assign(strategy_p1=s) for s in STRATEGIES], ignore_index=True)
    p1_inst = inst[inst["strategy"].astype(str).str.endswith("_p1")]
    unpaired = int(vs_direct["delta"].isna().sum() - p1_inst["relnaive_capped"].isna().sum())
    unpaired_run = int(vs_run["delta"].isna().sum() - p1_inst["relnaive_capped"].isna().sum())
    checks["G6_pairing"] = {"s3_instances": int(p1_inst["relnaive_capped"].notna().sum()),
                            "unpaired_with_direct": unpaired, "unpaired_with_run_stl": unpaired_run,
                            "passed": unpaired == 0 and unpaired_run == 0}
    passed = all(c["passed"] for c in checks.values())
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "checks.json", {"passed": passed, "checks": checks, "created_at_utc": utc_now()})
    if not passed:
        raise StageError(f"S3 checks failed: {sorted(k for k, v in checks.items() if not v['passed'])}")

    base_seed = main_seed
    by = ["model", "strategy_p1"]
    t_direct = one_sample(per_series(vs_direct, by), by, base_seed, "S3-vs-direct")   # > 0: p1 better than Direct
    t_run = one_sample(per_series(vs_run, by), by, base_seed, "S3-vs-run-stl")         # > 0: p1 better than the run's STL
    run_family = pd.concat([strategy_contrast(inst, "direct", s).assign(strategy_p1=s, specification="run")
                            for s in STRATEGIES], ignore_index=True)
    fam = pd.concat([vs_direct.assign(specification="p1"), run_family], ignore_index=True)
    t_family = one_sample(per_series(fam, ["strategy_p1", "specification"]), ["strategy_p1", "specification"],
                          base_seed, "S3-family")
    for t in (t_direct, t_run, t_family):
        t.rename(columns={"strategy_p1": "strategy"}, inplace=True)

    tables = {"s3_vs_direct": t_direct, "s3_vs_run_stl": t_run, "s3_family": t_family}
    (out / "tables").mkdir()
    for name, df in tables.items():
        df.to_csv(out / "tables" / f"{name}.csv", index=False)
    summary = {"status": "sensitivity analysis S3 (protocol change log v2.12 (e)); not a confirmatory hypothesis",
               "analysis_code_commit": commit, "created_at_utc": utc_now(),
               "merge_result_sha256": record["result_sha256"], "s3_configs_sha256": frozen["configs_sha256"],
               "tables": {name: sha256_json(df.to_dict(orient="list")) for name, df in tables.items()}}
    write_json(out / "s3.json", summary)
    return {"tests": len(t_direct) + len(t_run) + len(t_family)}


def main_specification(d: AnalysisData) -> AnalysisData:
    """Decision (1) of change log v2.12, the main specification of the paper: ARIMA and ETS under STL-SN
    and STL-AC are S3's forecasts with season length 1 on the targets without seasonality, and SARIMA,
    undefined there (decision (2)), leaves the STL strategies and keeps Direct. The run's analysis data
    with its statistical STL rows replaced; S3 must have been analysed with its checks passed."""
    checks = Path(d.run_dir) / ROOT / "analysis" / "checks.json"
    if not checks.exists() or not read_json(checks)["passed"]:
        raise AnalysisError("the main specification needs S3, analysed with its checks passed")
    tasks = [t for t in eval_tasks(d.tables["bucket_summary"], d.tables["samples"])
             if not t["task_id"].startswith(CONTROL_PREFIX)]
    p1, facts = s1._s1_rows(d.run_dir, tasks, ROOT, s1.ANALYSIS_COLUMNS)
    if facts["missing"] or facts["changed"]:
        raise AnalysisError(f"S3 rows missing or changed since its analysis: {facts['missing'] + facts['changed']}")

    def keep(df: pd.DataFrame) -> pd.DataFrame:
        stl = (df["family"].astype(str) == "statistical") & df["strategy"].astype(str).isin(STRATEGIES)
        return df[~stl]

    rows = pd.concat([keep(d.rows), p1[list(d.rows.columns)]], ignore_index=True)
    failed = keep(d.failed)
    return dataclasses.replace(d, rows=rows, failed=failed, instances=build_instances(rows, failed, d.tables["buckets"], d.main_seed))


def main(argv: list[str] | None = None) -> int:
    from src.utils import read_yaml

    parser = argparse.ArgumentParser(prog="python -m src.stages.sensitivity3", description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["freeze", "evaluate", "analyse"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard")
    parser.add_argument("--reference-s2", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = read_yaml(args.config)
    if args.stage == "evaluate" and args.data_dir is None:
        parser.error("evaluate needs --data-dir")
    if args.stage == "freeze":
        print(s1.run_freeze(args.run, study=S3, config=config))
    elif args.stage == "evaluate":
        print(s1.run_evaluate(args.run, config, args.data_dir, "cpu", DEFAULT_MANIFEST, workers=args.workers, study=S3,
                              shard=args.shard))
    else:
        print(run_analyse(args.run, config, DEFAULT_MANIFEST, args.reference_s2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
