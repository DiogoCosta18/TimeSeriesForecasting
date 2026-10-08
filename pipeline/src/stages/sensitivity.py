"""Sensitivity analysis S1, the STL configuration (protocol change log v2.2).

STL-SN under two STL variants, log STL and periodic STL, for the model of each family with
the lowest median RelNaive under Direct (ETS, XGBoost, NHITS, PatchTST); cohort scope, main
seed, every feature sample, both frequencies and the three windows of the run, against the
run's Direct forecasts (Direct is not recomputed). Everything lives in RUN/sensitivity_s1/:

    tune      one study per global model, frequency and variant on the variant's non-seasonal
              target, with the run's tuning settings (12 studies); --group cpu (XGBoost) or gpu
    freeze    the 12 entries -> configs_frozen.json with its content hash
    evaluate  --group gpu (NHITS, PatchTST) or cpu (XGBoost, ETS chunks; --workers N)
    analyse   the checks (as gates G1, G4, G5, G6, G9 for S1), then the 8 Wilcoxon tests with Holm,
              D26 labels, the run's own STL-SN beside them, medians by evolving-seasonality tercile

    python -m src.stages.sensitivity STAGE --config C --run RUN [--data-dir D] [--group G] [--workers N]

S1 adds no confirmatory hypothesis; the paper reports it as a sensitivity analysis.
"""
from __future__ import annotations

import argparse
import logging
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.data import ROW_COLUMNS, build_instances, per_series, strategy_contrast
from src.analysis.hypotheses import one_sample
from src.data.frozen import DEFAULT_MANIFEST, load_manifest, sha256_file
from src.data.load_m_datasets import load_dataset_pair
from src.data.schemas import FEATURE_NAMES
from src.forecast import engine, tuning
from src.forecast.engine import Provenance
from src.forecast.registry import config_key, family
from src.stages.io import (Heartbeat, StageError, TaskStore, code_commit, environment_lock_sha256, load_bundle, read_json,
                           sha256_json, stage_provenance, utc_now, write_json, write_parquet)
from src.stages.tasks import FREQUENCIES, STATISTICAL_CHUNK, Task, global_task_members, statistical_chunk_members
from src.validation.gates import load_merged

log = logging.getLogger(__name__)

ROOT = "sensitivity_s1"
S1_MODELS = {"statistical": "ETS", "ml": "XGBoost", "neural": "NHITS", "transformer": "PatchTST"}
GLOBAL_MODELS = ["XGBoost", "NHITS", "PatchTST"]
VARIANTS = ["log", "periodic"]
FROZEN_FORMAT = "rerun-s1-configs/1"
RULE_FEATURE = "feature_evolving_seasonality"


def group_of(model: str) -> str:
    return "gpu" if family(model) in ("neural", "transformer") else "cpu"


def entry_key(model: str, frequency: str, variant: str) -> str:
    return f"{model}|{frequency}|nonseasonal|{variant}"


# --- tasks ---------------------------------------------------------------------------------------------

def tune_tasks() -> list[dict]:
    return [{"task_id": f"s1tune|{model}|{frequency}|{variant}", "model": model, "frequency": frequency, "variant": variant,
             "group": group_of(model)}
            for model in GLOBAL_MODELS for frequency in FREQUENCIES for variant in VARIANTS]


def eval_tasks(bucket_summary: pd.DataFrame, samples: pd.DataFrame) -> list[dict]:
    features = [f for f in FEATURE_NAMES if f in set(bucket_summary["feature_name"])]
    tasks = []
    for frequency in FREQUENCIES:
        n_chunks = math.ceil(samples.loc[samples["frequency"] == frequency, "unique_id"].nunique() / STATISTICAL_CHUNK)
        for variant in VARIANTS:
            for feature in features:
                for model in GLOBAL_MODELS:
                    tasks.append({"task_id": f"s1eval|{feature}|{frequency}|{model}|{variant}", "kind": "global",
                                  "group": group_of(model), "frequency": frequency, "feature_name": feature, "model": model,
                                  "variant": variant})
            for chunk in range(n_chunks):
                tasks.append({"task_id": f"s1stat|ETS|{frequency}|{variant}|chunk{chunk:03d}", "kind": "statistical",
                              "group": "cpu", "frequency": frequency, "chunk": chunk, "model": "ETS", "variant": variant})
    return tasks


# --- R2 tune and freeze -----------------------------------------------------------------------------------

def run_tune(run_dir: Path, config: dict, data_dir: Path, group: str, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    run_dir = Path(run_dir)
    provenance = stage_provenance(load_bundle(run_dir, config, manifest_path), manifest_path)
    root = run_dir / ROOT / "tune"
    store = TaskStore(root, provenance)
    tasks = [t for t in tune_tasks() if t["group"] == group]
    if not tasks:
        raise StageError(f"no S1 tuning tasks in group {group!r}")
    prepare = run_dir / "prepare"
    tuning_set = pd.read_parquet(prepare / "tuning_set.parquet")
    cutoffs = pd.read_parquet(prepare / "cutoffs.parquet")
    storage = root / f"studies_{group}.sqlite"
    failed, done = [], 0
    with Heartbeat(root / f"heartbeat_{group}.json", "s1-tune", group) as beat:
        for frequency in FREQUENCIES:
            todo = [t for t in tasks if t["frequency"] == frequency and not store.is_done(t["task_id"], ".json")]
            done += sum(1 for t in tasks if t["frequency"] == frequency) - len(todo)
            if not todo:
                continue
            fcfg = config["frequencies"][frequency]
            m, h = int(fcfg["season_length"]), int(fcfg["horizon"])
            data = load_dataset_pair({"frequency": frequency, **fcfg}, data_dir, load_manifest(manifest_path))
            first_cutoff = cutoffs[(cutoffs["frequency"] == frequency) & (cutoffs["window"] == 0)].set_index("unique_id")["train_end_idx"]
            for task in todo:
                beat.update(task=task["task_id"], done=done, total=len(tasks), failed=len(failed))
                frame = tuning.tuning_frame(data, tuning_set, frequency, "nonseasonal", m, task["variant"])
                ends = frame.groupby("unique_id")["ds"].max()
                if not (ends == first_cutoff.loc[ends.index]).all():  # G7
                    raise StageError(f"{task['task_id']}: tuning data do not end at the first cutoff")
                start, attempts = time.perf_counter(), []
                for _ in range(2):
                    try:
                        entry, trials = tuning.run_study(task["model"], frequency, "nonseasonal", frame, h, m, config["tuning"],
                                                         int(config["random_seed"]))
                        break
                    except Exception as exc:  # D16: retried once, then listed
                        attempts.append(f"{type(exc).__name__}: {exc}")
                        log.error("%s failed:\n%s", task["task_id"], traceback.format_exc())
                else:
                    store.record_failure(task["task_id"], attempts)
                    failed.append(task["task_id"])
                    continue
                name = f"{tuning.study_name(task['model'], frequency, 'nonseasonal')}__{task['variant']}"
                tuning.archive_study(trials, storage, name)
                entry.update(study=f"{storage.name}:{name}", stl_variant=task["variant"], g7_tuning_end_equals_first_cutoff=True)
                write_json(store.output(task["task_id"], ".json"), entry)
                store.complete(task["task_id"], sha256_file(store.output(task["task_id"], ".json")),
                               {"failed_attempts": attempts, "compute_seconds": round(time.perf_counter() - start, 3)}, suffix=".json")
                store.clear_failure(task["task_id"])
                done += 1
        beat.update(task=None, done=done, total=len(tasks), failed=len(failed))
    if failed:
        raise StageError(f"{len(failed)} S1 studies failed twice: {failed}")
    return {"group": group, "studies": len(tasks)}


def run_freeze(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    target = run_dir / ROOT / "configs_frozen.json"
    if target.exists():
        raise StageError(f"{target} exists; frozen configurations are never overwritten")
    root = run_dir / ROOT / "tune"
    entries, provenances = {}, set()
    for task in tune_tasks():
        record_path = root / "tasks" / f"{task['task_id'].replace('|', '__')}.done.json"
        if not record_path.exists():
            raise StageError(f"S1 study {task['task_id']} has not completed")
        record = read_json(record_path)
        output = record_path.with_name(record["file"])
        if sha256_file(output) != record["sha256"]:
            raise StageError(f"{task['task_id']}: entry changed since it was recorded")
        entries[entry_key(task["model"], task["frequency"], task["variant"])] = read_json(output)
        provenances.add(tuple(sorted(record["provenance"].items())))
    if len(provenances) != 1:
        raise StageError(f"S1 studies were run with different provenance: {sorted(provenances)}")
    body = {"format": FROZEN_FORMAT, "provenance": dict(provenances.pop()), "entries": entries}
    digest = sha256_json(body)
    write_json(target, {**body, "configs_sha256": digest, "created_at_utc": utc_now()})
    return digest


def load_frozen(run_dir: Path) -> dict:
    record = read_json(Path(run_dir) / ROOT / "configs_frozen.json")
    body = {k: record[k] for k in ("format", "provenance", "entries")}
    if record.get("format") != FROZEN_FORMAT or sha256_json(body) != record.get("configs_sha256"):
        raise StageError("the S1 frozen configurations do not match their hash")
    return record


# --- R3 evaluate -------------------------------------------------------------------------------------------

def compute_task(job: dict) -> tuple[pd.DataFrame | None, list[str], float]:
    """One S1 task's rows, failed attempts and compute seconds (main process or worker)."""
    task, start = job["task"], time.perf_counter()
    if task["kind"] == "statistical":
        rows = pd.concat([engine.evaluate_statistical_series("ETS", task["frequency"], "stl_sn", series, job["cutoffs"], job["h"],
                                                             job["m"], job["prov"], stl_variant=task["variant"])
                          for series in job["series"]], ignore_index=True)
        return rows, [], time.perf_counter() - start
    attempts = []
    for _ in range(2):
        try:
            return engine.evaluate_global_task(job["spec"], job["pool"], job["cutoffs"], job["entries"], job["bucket_of"],
                                               job["h"], job["m"], job["prov"]), attempts, time.perf_counter() - start
        except Exception as exc:  # D16: retried once, then listed
            attempts.append(f"{type(exc).__name__}: {exc}")
            log.error("%s failed:\n%s", task["task_id"], traceback.format_exc())
    return None, attempts, time.perf_counter() - start


def s1_provenance(run_dir: Path, bundle: dict, frozen: dict, manifest_path: Path) -> Provenance:
    prov = Provenance(code_commit=code_commit(), environment_lock_sha256=environment_lock_sha256(),
                      data_manifest_sha256=sha256_file(manifest_path), bundle_sha256=bundle["bundle_sha256"],
                      configs_sha256=frozen["configs_sha256"])
    for field in ("code_commit", "environment_lock_sha256", "data_manifest_sha256", "bundle_sha256"):
        if getattr(prov, field) != frozen["provenance"][field]:
            raise StageError(f"S1 evaluation runs with the {field} its configurations were tuned with")
    return prov


def run_evaluate(run_dir: Path, config: dict, data_dir: Path, group: str, manifest_path: Path = DEFAULT_MANIFEST,
                 workers: int = 1) -> dict:
    run_dir = Path(run_dir)
    if workers > 1 and group != "cpu":
        raise StageError("--workers is for the cpu group; the gpu group trains one task at a time")
    bundle = load_bundle(run_dir, config, manifest_path)
    frozen = load_frozen(run_dir)
    prov = s1_provenance(run_dir, bundle, frozen, manifest_path)
    prepare = run_dir / "prepare"
    samples = pd.read_parquet(prepare / "samples.parquet")
    buckets = pd.read_parquet(prepare / "buckets.parquet")
    tasks = [t for t in eval_tasks(pd.read_parquet(prepare / "bucket_summary.parquet"), samples) if t["group"] == group]
    if not tasks:
        raise StageError(f"no S1 evaluation tasks in group {group!r}")
    root = run_dir / ROOT / "evaluate"
    store = TaskStore(root, asdict(prov))
    cutoffs_all = pd.read_parquet(prepare / "cutoffs.parquet")
    main_seed = int(config["random_seed"])
    failed, finished = [], 0

    def finish(task, rows, attempts, seconds) -> None:
        nonlocal finished
        finished += 1
        if rows is None:
            store.record_failure(task["task_id"], attempts)
            failed.append(task["task_id"])
        else:
            rows.insert(0, "task_id", task["task_id"])
            sha = write_parquet(store.output(task["task_id"]), rows)
            store.complete(task["task_id"], sha, {"rows": int(len(rows)), "failed_rows": int((rows["status"] != "trained").sum()),
                                                  "failed_attempts": attempts, "compute_seconds": round(seconds, 3)})
            store.clear_failure(task["task_id"])
        beat.update(task=task["task_id"], done=finished, total=len(tasks), failed=len(failed))

    with Heartbeat(root / f"heartbeat_{group}.json", "s1-evaluate", group) as beat:
        finished = sum(1 for t in tasks if store.is_done(t["task_id"]))
        for frequency in FREQUENCIES:
            pending = [t for t in tasks if t["frequency"] == frequency and not store.is_done(t["task_id"])]
            if not pending:
                continue
            fcfg = config["frequencies"][frequency]
            m, h = int(fcfg["season_length"]), int(fcfg["horizon"])
            union = set(samples.loc[samples["frequency"] == frequency, "unique_id"])
            data = load_dataset_pair({"frequency": frequency, **fcfg}, data_dir, load_manifest(manifest_path))
            data = data[data["unique_id"].isin(union)][["unique_id", "source_dataset", "t", "y"]]
            by_id = {uid: g for uid, g in data.groupby("unique_id", sort=True)}
            cutoffs = cutoffs_all[cutoffs_all["frequency"] == frequency]

            def job(task) -> dict:
                common = {"task": task, "h": h, "m": m, "prov": prov}
                if task["kind"] == "statistical":
                    members = statistical_chunk_members(samples, frequency, task["chunk"])
                    return {**common, "series": [by_id[u] for u in members], "cutoffs": cutoffs[cutoffs["unique_id"].isin(members)]}
                spec = Task("global", frequency, task["model"], "stl_sn", None, task["feature_name"], "cohort", main_seed, 0)
                members, bucket_of = global_task_members(spec, samples, buckets)
                entry = frozen["entries"][entry_key(task["model"], frequency, task["variant"])]
                return {**common, "pool": pd.concat([by_id[u] for u in members], ignore_index=True), "bucket_of": bucket_of,
                        "cutoffs": cutoffs[cutoffs["unique_id"].isin(members)],
                        "entries": {config_key(task["model"], frequency, "nonseasonal"): entry},
                        "spec": {"feature_name": task["feature_name"], "frequency": frequency, "strategy": "stl_sn",
                                 "model": task["model"], "scope": "cohort", "seed": main_seed, "stl_variant": task["variant"]}}

            if workers <= 1:
                for task in pending:
                    beat.update(task=task["task_id"])
                    finish(task, *compute_task(job(task)))
            else:
                with ProcessPoolExecutor(workers) as pool:
                    futures = {pool.submit(compute_task, job(task)): task for task in pending}
                    for future in as_completed(futures):
                        finish(futures[future], *future.result())
        beat.update(task=None, done=finished, total=len(tasks), failed=len(failed))
    if failed:
        raise StageError(f"{len(failed)} S1 tasks failed twice: {failed}")
    return {"group": group, "tasks": len(tasks)}


# --- analyse ---------------------------------------------------------------------------------------------

def _s1_rows(run_dir: Path, tasks: list[dict]) -> tuple[pd.DataFrame, dict]:
    """The S1 rows of every completed task (hash-checked) and the facts the checks need."""
    root = Path(run_dir) / ROOT / "evaluate"
    frames, missing, changed, commits, pins, configs = [], [], [], set(), set(), set()
    for task in tasks:
        record_path = root / "tasks" / f"{task['task_id'].replace('|', '__')}.done.json"
        if not record_path.exists():
            missing.append(task["task_id"])
            continue
        record = read_json(record_path)
        output = record_path.with_name(record["file"])
        if sha256_file(output) != record["sha256"]:
            changed.append(task["task_id"])
            continue
        frames.append(pd.read_parquet(output))
        commits.add(record["provenance"]["code_commit"])
        configs.add(record["provenance"]["configs_sha256"])
        pins.add(bool(record["platform"]["pinned"]))
    listed = sorted(p.name for p in (root / "failures").glob("*.json")) if (root / "failures").exists() else []
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return rows, {"missing": missing, "changed": changed, "commits": sorted(commits), "pins": sorted(pins),
                  "configs": sorted(configs), "listed_failures": listed}


def run_analyse(run_dir: Path, config: dict, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    run_dir = Path(run_dir)
    out = run_dir / ROOT / "analysis"
    if out.exists() and any(out.iterdir()):
        raise StageError(f"{out} is not empty; S1 outputs are never overwritten")
    commit = code_commit()
    bundle = load_bundle(run_dir, config, manifest_path)
    frozen = load_frozen(run_dir)
    prepare = run_dir / "prepare"
    samples, buckets = pd.read_parquet(prepare / "samples.parquet"), pd.read_parquet(prepare / "buckets.parquet")
    bucket_summary, features = pd.read_parquet(prepare / "bucket_summary.parquet"), pd.read_parquet(prepare / "features.parquet")
    tasks = eval_tasks(bucket_summary, samples)
    s1, facts = _s1_rows(run_dir, tasks)
    record, rows, failed = load_merged(run_dir, ROW_COLUMNS)          # the run's merged rows, hash-checked
    main_seed = int(config["random_seed"])
    models = list(S1_MODELS.values())
    keep = rows["model"].isin(models) & (rows["scope"] == "cohort") & rows["strategy"].isin(["direct", "stl_sn"])
    run_rows = rows[keep & (rows["seed"].isna() | (rows["seed"] == main_seed))]

    checks = {"G5_complete": {"expected": len(tasks), "missing": facts["missing"], "changed": facts["changed"]},
              "G4_commit_and_pins": {"commits": facts["commits"], "pinned": facts["pins"], "configs": facts["configs"],
                                     "frozen_configs": frozen["configs_sha256"],
                                     "bundle_matches_run": bundle["bundle_sha256"] == record["provenance"]["bundle_sha256"]}}
    checks["G5_complete"]["passed"] = not facts["missing"] and not facts["changed"]
    checks["G4_commit_and_pins"]["passed"] = (len(facts["commits"]) == 1 and facts["pins"] == [True]
                                              and facts["configs"] == [frozen["configs_sha256"]]
                                              and checks["G4_commit_and_pins"]["bundle_matches_run"])
    trained = s1[s1["status"] == "trained"] if len(s1) else s1
    metrics = ["relnaive_capped", "relnaive", "mase", "smape", "mae", "pocid"]
    nonfinite = int((~np.isfinite(trained[metrics].to_numpy(dtype=float))).any(axis=1).sum()) if len(trained) else 0
    checks["G1_trained_rows"] = {"rows": int(len(s1)), "failed_rows": int(len(s1) - len(trained)),
                                 "listed_task_failures": facts["listed_failures"], "passed": not facts["listed_failures"]}
    checks["G9_finite"] = {"non_finite_rows": nonfinite, "passed": nonfinite == 0}

    s1_inst_rows = s1.assign(strategy="stl_sn_" + s1["stl_variant"].astype(str)).drop(columns="stl_variant") if len(s1) else s1
    both = pd.concat([run_rows, s1_inst_rows[run_rows.columns]], ignore_index=True)
    inst = build_instances(both, failed.iloc[0:0], buckets, main_seed)
    inst = inst[inst["main_seed"]]
    deltas = []
    for variant in VARIANTS:
        d = strategy_contrast(inst, "direct", f"stl_sn_{variant}").assign(variant=variant)
        deltas.append(d)
    d1 = pd.concat(deltas, ignore_index=True)
    s1_present = inst[inst["strategy"].astype(str).str.startswith("stl_sn_") & inst["relnaive_capped"].notna()]
    unpaired = int(d1["delta"].isna().sum() - (inst[inst["strategy"].astype(str).str.startswith("stl_sn_")]["relnaive_capped"].isna().sum()))
    checks["G6_pairing"] = {"s1_instances": int(len(s1_present)), "unpaired_with_direct": unpaired, "passed": unpaired == 0}
    passed = all(c["passed"] for c in checks.values())
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "checks.json", {"passed": passed, "checks": checks, "created_at_utc": utc_now()})
    if not passed:
        raise StageError(f"S1 checks failed: {sorted(k for k, v in checks.items() if not v['passed'])}")

    base_seed = int(config["random_seed"])
    tests = one_sample(per_series(d1, ["model", "variant"]), ["model", "variant"], base_seed, "S1")
    default = one_sample(per_series(strategy_contrast(inst, "direct", "stl_sn"), ["model"]), ["model"], base_seed, "S1-default")
    gains = tests[tests["label"] == "gain"][["model", "variant"]].to_dict(orient="records")
    decision = ("the STL conclusion stands: no model gains under either variant" if not gains else
                f"the effect of STL depends on the configuration for {gains}")

    cuts = bucket_summary[bucket_summary["feature_name"] == RULE_FEATURE].set_index("frequency")
    f = features[features["eligible"]][["unique_id", "frequency", RULE_FEATURE]].copy()
    c1, c2 = f["frequency"].map(cuts["tercile_cut_1"]), f["frequency"].map(cuts["tercile_cut_2"])
    f["tercile"] = np.select([f[RULE_FEATURE] <= c1, f[RULE_FEATURE] <= c2], ["Low", "Medium"], "High")
    all_deltas = pd.concat([d1, strategy_contrast(inst, "direct", "stl_sn").assign(variant="default")], ignore_index=True)
    ps = per_series(all_deltas, ["model", "variant"]).astype({"unique_id": str})
    ps = ps.merge(f[["unique_id", "tercile"]], on="unique_id", how="left")
    if ps["tercile"].isna().any():
        raise StageError("series without an evolving-seasonality value")
    terciles = (ps.groupby(["model", "variant", "tercile"], observed=True)["delta"]
                .agg(n_series="size", median="median", share_positive=lambda x: float((x > 0).mean())).reset_index())

    tables = {"s1_tests": tests, "s1_default": default, "s1_evolving_terciles": terciles}
    (out / "tables").mkdir()
    for name, df in tables.items():
        df.to_csv(out / "tables" / f"{name}.csv", index=False)
    summary = {"status": "sensitivity analysis S1 (protocol change log v2.2); not a confirmatory hypothesis",
               "analysis_code_commit": commit, "created_at_utc": utc_now(), "decision": decision, "gains": gains,
               "merge_result_sha256": record["result_sha256"], "s1_configs_sha256": frozen["configs_sha256"],
               "tables": {name: sha256_json(df.to_dict(orient="list")) for name, df in tables.items()}}
    write_json(out / "s1.json", summary)
    return {"decision": decision, "tests": len(tests)}


def main(argv: list[str] | None = None) -> int:
    from src.utils import read_yaml

    parser = argparse.ArgumentParser(prog="python -m src.stages.sensitivity", description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["tune", "freeze", "evaluate", "analyse"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--group", choices=["cpu", "gpu"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = read_yaml(args.config)
    if args.stage in ("tune", "evaluate") and (args.group is None or args.data_dir is None):
        parser.error(f"{args.stage} needs --group and --data-dir")
    if args.stage == "tune":
        print(run_tune(args.run, config, args.data_dir, args.group, DEFAULT_MANIFEST))
    elif args.stage == "freeze":
        print(run_freeze(args.run))
    elif args.stage == "evaluate":
        print(run_evaluate(args.run, config, args.data_dir, args.group, DEFAULT_MANIFEST, workers=args.workers))
    else:
        print(run_analyse(args.run, config, DEFAULT_MANIFEST))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
