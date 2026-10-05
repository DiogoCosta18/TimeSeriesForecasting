"""Stage R4, merge (protocol Section 9.1; D16, D18-D20; tests U14, U15).

Collects every completed evaluation task of every shard into the result bundle
``merged/``:
- ``rows.parquet``: the trained rows (keys, metrics, timing, forecast hash, provenance);
  the forecast vectors stay in the shard outputs, whose hashes the record lists;
- ``failed_rows.parquet``: the statistical per-series failures (D16);
- ``merge.json``: the provenance, every task output with its SHA-256, the expected grid,
  missing tasks, listed and resolved failures, D16 reruns, and the bundle's hash.

The merge refuses, and writes nothing, when
- a shard output is missing, changed since its record, or not a task of the grid;
- any provenance differs from the run's: data manifest, prepare bundle, frozen
  configurations, environment lock or code commit (U15); a later commit is accepted
  only for a task in its shard's D16 register;
- a tuning study no longer matches its frozen configuration;
- rows disagree with their task (keys, members, windows, provenance), their forecasts
  do not match their hashes, or a trained row lacks a positive training time;
- a tercile row lacks exactly one cohort twin in the same bucket (U14).
An incomplete grid is merged and reported, so gate G5 fails rather than the merge.
A bundle whose gates passed is frozen and never rebuilt.
"""
from __future__ import annotations

import shutil
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.frozen import DEFAULT_MANIFEST, sha256_file
from src.forecast.engine import Provenance
from src.forecast.registry import config_key, family
from src.forecast.result import forecast_hash
from src.forecast.tuning import load_frozen_configs
from src.stages.io import StageError, code_commit, load_bundle, read_json, safe_name, sha256_json, utc_now, write_json, write_parquet
from src.stages.tasks import BUCKET_SCOPE, Task, evaluation_tasks, global_task_members, grid_counts, statistical_chunk_members, tuning_tasks

MERGE_FORMAT = "rerun-v2-merge-1"
VECTORS = ["y_test", "yhat", "yhat_naive"]
PROVENANCE_FIELDS = [f.name for f in fields(Provenance)]
TWIN_KEYS = ["feature_name", "frequency", "strategy", "model", "seed", "unique_id", "window"]


class MergeError(StageError):
    """The shard outputs cannot be merged without mixing or losing results."""


def run_provenance(run_dir: Path, bundle: dict, manifest_path: Path) -> tuple[dict, dict]:
    """The provenance every row must carry (D19) and the frozen configurations."""
    frozen = load_frozen_configs(Path(run_dir) / "configs_frozen.json")
    fp = frozen["provenance"]
    main = {"code_commit": fp["code_commit"], "environment_lock_sha256": fp["environment_lock_sha256"],
            "data_manifest_sha256": sha256_file(manifest_path), "bundle_sha256": bundle["bundle_sha256"],
            "configs_sha256": frozen["configs_sha256"]}
    if fp["data_manifest_sha256"] != main["data_manifest_sha256"] or fp["bundle_sha256"] != main["bundle_sha256"]:
        raise MergeError("the frozen configurations were tuned on other data or another prepare bundle")
    return main, frozen


def _platform(task_id: str, record: dict) -> dict:
    platform = record.get("platform")
    if not platform:
        raise MergeError(f"{task_id}: completion record without its platform")
    return {k: platform[k] for k in ("cpu_model", "libc", "numba_cpu_name", "pinned")}


def _check_studies(run_dir: Path, frozen: dict) -> dict:
    platforms = {}
    for task in tuning_tasks():
        record_path = run_dir / "tune" / task.shard / "tasks" / f"{safe_name(task.task_id)}.done.json"
        if not record_path.exists():
            raise MergeError(f"study {task.task_id} has no completion record")
        record = read_json(record_path)
        if record["provenance"] != frozen["provenance"]:
            raise MergeError(f"study {task.task_id} carries other provenance than the frozen configurations")
        output = record_path.with_name(record["file"])
        if not output.exists() or sha256_file(output) != record["sha256"]:
            raise MergeError(f"study {task.task_id}: output missing or changed")
        if read_json(output) != frozen["entries"][config_key(task.model, task.frequency, task.target)]:
            raise MergeError(f"study {task.task_id} does not match its frozen configuration")
        platforms[task.task_id] = _platform(task.task_id, record)
    return platforms


def _check_provenance(task_id: str, prov: dict, main: dict, d16: dict) -> None:
    if set(prov) != set(main):
        raise MergeError(f"{task_id}: provenance fields {sorted(prov)} differ from {sorted(main)}")
    for field in main:
        if field != "code_commit" and prov[field] != main[field]:
            raise MergeError(f"{task_id}: {field} {prov[field]} differs from the run's {main[field]} (U15)")
    if prov["code_commit"] != main["code_commit"]:
        entry = d16.get(task_id)
        if entry is None or entry["fix_commit"] != prov["code_commit"]:
            raise MergeError(f"{task_id}: code commit {prov['code_commit']} is not the run's and not a registered D16 rerun")


def _expected_pairs(task: Task, samples: pd.DataFrame, buckets: pd.DataFrame, n_windows: int) -> set:
    if task.kind == "statistical":
        members = statistical_chunk_members(samples, task.frequency, task.chunk)
    else:
        members, _ = global_task_members(task, samples, buckets)
    return {(uid, w) for uid in members for w in range(n_windows)}


def _check_rows(task: Task, rows: pd.DataFrame, record: dict, expected_pairs: set, h: int) -> None:
    tid = task.task_id
    if len(rows) != record["rows"] or (rows["task_id"] != tid).any():
        raise MergeError(f"{tid}: rows do not match the completion record")
    statistical = task.kind == "statistical"
    keys = {"frequency": task.frequency, "strategy": task.strategy, "model": task.model, "scope": task.scope,
            "family": family(task.model)}
    for column, value in keys.items():
        if (rows[column] != value).any():
            raise MergeError(f"{tid}: column {column} disagrees with the task")
    if statistical:
        if rows["feature_name"].notna().any() or rows["seed"].notna().any() or rows["bucket"].notna().any():
            raise MergeError(f"{tid}: statistical rows carry feature, seed or bucket")
    elif (rows["feature_name"] != task.feature_name).any() or (rows["seed"] != task.seed).any():
        raise MergeError(f"{tid}: feature or seed disagrees with the task")
    pairs = list(zip(rows["unique_id"], rows["window"].astype(int)))
    if len(set(pairs)) != len(pairs) or set(pairs) != expected_pairs:
        raise MergeError(f"{tid}: series and windows differ from the task's members")
    for field in PROVENANCE_FIELDS:
        if (rows[field] != record["provenance"][field]).any():
            raise MergeError(f"{tid}: rows carry other {field} than their completion record")
    status = rows["status"]
    if not status.isin(["trained", "failed"]).all() or (not statistical and (status != "trained").any()):
        raise MergeError(f"{tid}: unexpected row status")
    for yhat, y_test, naive, digest in rows.loc[status == "trained", ["yhat", "y_test", "yhat_naive", "forecast_hash"]].itertuples(index=False):
        values = np.asarray(yhat, dtype=float)
        if len(values) != h or len(y_test) != h or len(naive) != h or not np.isfinite(values).all():
            raise MergeError(f"{tid}: a forecast is incomplete")
        if forecast_hash(values) != digest:
            raise MergeError(f"{tid}: a forecast does not match its hash")
    if rows.loc[status == "failed", "failure"].isna().any():
        raise MergeError(f"{tid}: a failed row has no reason")
    times = rows.loc[status == "trained", ["fit_seconds", "predict_seconds"]].to_numpy(dtype=float)
    if not (np.isfinite(times).all() and (times[:, 0] > 0).all() and (times[:, 1] >= 0).all()):
        raise MergeError(f"{tid}: a trained row has a missing or non-positive training time (A9, G10)")


def check_twins(rows: pd.DataFrame) -> dict:
    """U14 / G6: every tercile row has exactly one cohort twin in the same bucket."""
    glob = rows[(rows["family"] != "statistical") & (rows["status"] == "trained")]
    cohort = glob[glob["scope"] == "cohort"][TWIN_KEYS + ["bucket"]]
    tercile = glob[glob["scope"] != "cohort"][TWIN_KEYS + ["bucket", "scope"]]
    wrong_bucket = int((tercile["bucket"].map(BUCKET_SCOPE) != tercile["scope"]).sum())
    duplicated = int(cohort.duplicated(TWIN_KEYS).sum())
    paired = tercile.merge(cohort, on=TWIN_KEYS, how="left", suffixes=("", "_cohort"), indicator=True)
    missing = int((paired["_merge"] == "left_only").sum())
    other_bucket = int(((paired["_merge"] == "both") & (paired["bucket"] != paired["bucket_cohort"])).sum())
    return {"tercile_rows": int(len(tercile)), "missing_twin": missing, "duplicated_cohort_rows": duplicated,
            "bucket_not_scope": wrong_bucket, "twin_in_other_bucket": other_bucket,
            "passed": missing == duplicated == wrong_bucket == other_bucket == 0}


def run_merge(run_dir: Path, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    run_dir = Path(run_dir)
    out = run_dir / "merged"
    if (out / "gates.json").exists() and read_json(out / "gates.json").get("passed"):
        raise MergeError(f"{out} passed its gates; the result bundle is frozen")
    config = read_json(run_dir / "prepare" / "bundle.json")["config"]
    bundle = load_bundle(run_dir, config, manifest_path)
    main, frozen = run_provenance(run_dir, bundle, manifest_path)
    study_platforms = _check_studies(run_dir, frozen)
    prepare = run_dir / "prepare"
    samples = pd.read_parquet(prepare / "samples.parquet")
    buckets = pd.read_parquet(prepare / "buckets.parquet")
    grid = evaluation_tasks(pd.read_parquet(prepare / "bucket_summary.parquet"), samples, int(config["random_seed"]),
                            list(config["seed_check_seeds"]))
    expected = {t.task_id: t for t in grid}
    n_windows = int(config["validation"]["n_windows"])

    tasks, d16, listed, resolved, frames = {}, {}, {}, {}, []
    shard_dirs = sorted(p for p in (run_dir / "evaluate").glob("*") if p.is_dir()) if (run_dir / "evaluate").exists() else []
    for shard_dir in shard_dirs:
        register = read_json(shard_dir / "d16_reruns.json") if (shard_dir / "d16_reruns.json").exists() else {}
        for task_id, entry in register.items():
            if entry["main_commit"] != main["code_commit"]:
                raise MergeError(f"{task_id}: D16 rerun registered against another main commit")
        d16.update(register)
        for folder, target in ((shard_dir / "failures", listed), (shard_dir / "failures" / "resolved", resolved)):
            for path in sorted(folder.glob("*.json")):
                failure = read_json(path)
                target[failure["task_id"]] = failure
        for record_path in sorted((shard_dir / "tasks").glob("*.done.json")):
            record = read_json(record_path)
            task_id = record["task_id"]
            task = expected.get(task_id)
            if task is None or task.shard != shard_dir.name or record_path.name != f"{safe_name(task_id)}.done.json":
                raise MergeError(f"{record_path}: not a task of shard {shard_dir.name} in the grid")
            output = record_path.with_name(record["file"])
            if not output.exists() or sha256_file(output) != record["sha256"]:
                raise MergeError(f"{task_id}: output missing or changed since it was recorded")
            _check_provenance(task_id, record["provenance"], main, d16)
            rows = pd.read_parquet(output)
            fcfg = config["frequencies"][task.frequency]
            _check_rows(task, rows, record, _expected_pairs(task, samples, buckets, n_windows), int(fcfg["horizon"]))
            frames.append(rows.drop(columns=VECTORS))
            tasks[task_id] = {"shard": shard_dir.name, "file": f"evaluate/{shard_dir.name}/tasks/{record['file']}",
                              "sha256": record["sha256"], "rows": record["rows"], "failed_rows": record["failed_rows"],
                              "failed_attempts": record.get("failed_attempts", []),
                              "code_commit": record["provenance"]["code_commit"], "platform": _platform(task_id, record)}
    if set(listed) & set(tasks):
        raise MergeError(f"tasks both completed and listed as failed: {sorted(set(listed) & set(tasks))}")
    if set(d16) - set(tasks):
        raise MergeError(f"D16 reruns without a completed task: {sorted(set(d16) - set(tasks))}")
    if not frames:
        raise MergeError("no completed evaluation tasks to merge")

    rows = pd.concat(frames, ignore_index=True)
    rows["seed"] = rows["seed"].astype("Int64")
    rows = rows.sort_values(["task_id", "unique_id", "window"], kind="mergesort").reset_index(drop=True)
    twins = check_twins(rows)
    if not twins["passed"]:
        raise MergeError(f"tercile rows without exactly one cohort twin (U14): {twins}")

    if out.exists():
        shutil.rmtree(out)
    trained, failed = rows[rows["status"] == "trained"].reset_index(drop=True), rows[rows["status"] == "failed"].reset_index(drop=True)
    files = {"rows": write_parquet(out / "rows.parquet", trained), "failed_rows": write_parquet(out / "failed_rows.parquet", failed)}
    missing = sorted(set(expected) - set(tasks) - set(listed))
    body = {
        "format": MERGE_FORMAT,
        "provenance": main,
        "config_sha256": bundle["config_sha256"],
        "grid": {"expected": len(expected), "completed": len(tasks), "listed_failures": sorted(listed), "missing": missing,
                 "counts": grid_counts(grid)},
        "tasks": tasks,
        "study_platforms": study_platforms,
        "d16_reruns": d16,
        "failures": {"listed": listed, "resolved": resolved},
        "files": {name: {"file": f"{name}.parquet", "sha256": sha} for name, sha in files.items()},
        "rows": {"trained": int(len(trained)), "failed": int(len(failed))},
        "twins": twins,
    }
    record = {**body, "result_sha256": sha256_json(body), "created_at_utc": utc_now(), "merge_code_commit": code_commit()}
    write_json(out / "merge.json", record)
    return {"result_sha256": record["result_sha256"], "completed": len(tasks), "expected": len(expected),
            "missing": len(missing), "listed_failures": len(listed), "rows": record["rows"]}
