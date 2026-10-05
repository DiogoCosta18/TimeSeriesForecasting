"""Stage R3, evaluate (per shard) of protocol Section 9.1.

Runs the shard's tasks in priority order (cohort, tercile, seed check). A finished task
(completion record and output agreeing with the current provenance) is skipped, so an
interrupted shard resumes where it stopped and produces the same outputs (test I4).
A global task that raises is retried once; a second failure is listed in
``failures/`` and the stage ends with an error after running the remaining tasks
(D16). Statistical chunks record per-series failures as failed rows (D16).

Evaluation runs at the code commit, environment and data manifest the configurations
were frozen with (R0). Another commit is accepted only to rerun listed failures after a
committed fix (D16, ``only``): the fix must descend from the frozen commit and nothing
else may differ. Each such rerun is entered in the shard's ``d16_reruns.json``, which
the merge needs to accept that task's commit (G4). Resolved failures are kept in
``failures/resolved/`` for the failure accounting (A8).
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.data.frozen import DEFAULT_MANIFEST, load_manifest
from src.data.load_m_datasets import load_dataset_pair
from src.forecast import engine, tuning
from src.forecast.engine import Provenance
from src.stages.io import (Heartbeat, StageError, TaskStore, evaluation_provenance, is_ancestor, load_bundle, read_json,
                           utc_now, write_json, write_parquet)
from src.stages.tasks import Task, evaluation_tasks, global_task_members, statistical_chunk_members

log = logging.getLogger(__name__)

D16_REGISTER = "d16_reruns.json"


def check_commit(prov: Provenance, frozen_provenance: dict, store: TaskStore, tasks: list[Task], only) -> bool:
    """Whether this is a D16 rerun at a fix commit; refuses any other departure from R0."""
    for field in ("environment_lock_sha256", "data_manifest_sha256"):
        if getattr(prov, field) != frozen_provenance[field]:
            raise StageError(f"{field} differs from the one the configurations were tuned with")
    main = frozen_provenance["code_commit"]
    if prov.code_commit == main:
        return False
    if only is None:
        raise StageError(f"evaluation runs at commit {main}, the frozen configurations'; another commit only "
                         "reruns listed failures after a committed fix (D16, --only)")
    if not is_ancestor(main, prov.code_commit):
        raise StageError(f"commit {prov.code_commit} does not descend from {main}")
    current = {k: v for k, v in asdict(prov).items() if k != "code_commit"}
    for task in tasks:
        failure = store.failure(task.task_id)
        if failure is None:
            raise StageError(f"{task.task_id} is not a listed failure; after a fix only listed failures are rerun (D16)")
        if {k: v for k, v in failure["provenance"].items() if k != "code_commit"} != current:
            raise StageError(f"{task.task_id} failed with other inputs than the current ones")
    return True


def _resolve_failure(store: TaskStore, task_id: str, register_path: Path | None, main_commit: str) -> None:
    failure = store.failure(task_id)
    if failure is None:
        return
    resolved = {**failure, "resolved_at_utc": utc_now(), "resolved_by_commit": store.provenance["code_commit"]}
    write_json(store.root / "failures" / "resolved" / store.failure_path(task_id).name, resolved)
    if register_path is not None:
        register = read_json(register_path) if register_path.exists() else {}
        register[task_id] = {"main_commit": main_commit, "fix_commit": store.provenance["code_commit"],
                             "failure": failure, "rerun_at_utc": resolved["resolved_at_utc"]}
        write_json(register_path, register)
    store.clear_failure(task_id)


def run_evaluate(run_dir: Path, config: dict, data_dir: Path, shard: str, manifest_path: Path = DEFAULT_MANIFEST,
                 only: set[str] | None = None) -> dict:
    """Evaluate a shard; ``only`` restricts it to the given task ids (a rerun after a fix, D16)."""
    run_dir = Path(run_dir)
    bundle = load_bundle(run_dir, config, manifest_path)
    prov = evaluation_provenance(run_dir, bundle, manifest_path)
    frozen = tuning.load_frozen_configs(run_dir / "configs_frozen.json")
    if frozen["provenance"].get("bundle_sha256") != bundle["bundle_sha256"]:
        raise StageError("the frozen configurations were tuned on another prepare bundle")
    prepare = run_dir / "prepare"
    samples = pd.read_parquet(prepare / "samples.parquet")
    buckets = pd.read_parquet(prepare / "buckets.parquet")
    bucket_summary = pd.read_parquet(prepare / "bucket_summary.parquet")
    tasks = [t for t in evaluation_tasks(bucket_summary, samples, int(config["random_seed"]), list(config["seed_check_seeds"]))
             if t.shard == shard and (only is None or t.task_id in only)]
    if not tasks:
        raise StageError(f"no evaluation tasks in shard {shard!r}")
    if only is not None and len(tasks) != len(only):
        raise StageError(f"not tasks of shard {shard!r}: {sorted(set(only) - {t.task_id for t in tasks})}")

    root = run_dir / "evaluate" / shard
    store = TaskStore(root, asdict(prov))
    d16 = check_commit(prov, frozen["provenance"], store, tasks, only)
    register_path = root / D16_REGISTER if d16 else None

    frequency = tasks[0].frequency
    fcfg = config["frequencies"][frequency]
    m, h = int(fcfg["season_length"]), int(fcfg["horizon"])
    union = set(samples.loc[samples["frequency"] == frequency, "unique_id"])
    data = load_dataset_pair({"frequency": frequency, **fcfg}, data_dir, load_manifest(manifest_path))
    data = data[data["unique_id"].isin(union)][["unique_id", "source_dataset", "t", "y"]]
    by_id = {uid: g for uid, g in data.groupby("unique_id", sort=True)}
    cutoffs = pd.read_parquet(prepare / "cutoffs.parquet")
    cutoffs = cutoffs[cutoffs["frequency"] == frequency]

    failed = []
    with Heartbeat(root / "heartbeat.json", "evaluate", shard) as beat:
        for i, task in enumerate(tasks):
            beat.update(task=task.task_id, done=i, total=len(tasks), failed=len(failed))
            if store.is_done(task.task_id):
                continue
            attempts = []
            if task.kind == "statistical":
                members = statistical_chunk_members(samples, frequency, task.chunk)
                rows = pd.concat([engine.evaluate_statistical_series(task.model, frequency, task.strategy, by_id[uid],
                                                                     cutoffs, h, m, prov) for uid in members],
                                 ignore_index=True)
            else:
                members, bucket_of = global_task_members(task, samples, buckets)
                spec = {"feature_name": task.feature_name, "frequency": frequency, "strategy": task.strategy,
                        "model": task.model, "scope": task.scope, "seed": task.seed}
                pool = pd.concat([by_id[uid] for uid in members], ignore_index=True)
                for _ in range(2):
                    try:
                        rows = engine.evaluate_global_task(spec, pool, cutoffs, frozen["entries"], bucket_of, h, m, prov)
                        break
                    except Exception as exc:  # D16: retried once, then listed
                        attempts.append(f"{type(exc).__name__}: {exc}")
                        log.error("%s failed:\n%s", task.task_id, traceback.format_exc())
                else:
                    store.record_failure(task.task_id, attempts)
                    failed.append(task.task_id)
                    continue
            rows.insert(0, "task_id", task.task_id)
            sha = write_parquet(store.output(task.task_id), rows)
            store.complete(task.task_id, sha, {"rows": int(len(rows)), "failed_rows": int((rows["status"] != "trained").sum()),
                                               "failed_attempts": attempts})
            _resolve_failure(store, task.task_id, register_path, frozen["provenance"]["code_commit"])
        beat.update(task=None, done=len(tasks), failed=len(failed))
    if failed:
        raise StageError(f"{len(failed)} global tasks failed twice: {failed}")
    return {"shard": shard, "tasks": len(tasks)}
