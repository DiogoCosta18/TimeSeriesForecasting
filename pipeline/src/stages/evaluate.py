"""Stage R3, evaluate (per shard) of protocol Section 9.1.

Runs the shard's tasks in priority order (cohort, tercile, seed check). A finished task
(completion record and output agreeing with the current provenance) is skipped, so an
interrupted shard resumes where it stopped and produces the same outputs (test I4).
A global task that raises is retried once; a second failure is listed in
``failures/`` and the stage ends with an error after running the remaining tasks
(D16). Statistical chunks record per-series failures as failed rows (D16).
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
from src.stages.io import Heartbeat, StageError, TaskStore, evaluation_provenance, load_bundle, write_parquet
from src.stages.tasks import evaluation_tasks, global_task_members, statistical_chunk_members

log = logging.getLogger(__name__)


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
    frequency = tasks[0].frequency
    fcfg = config["frequencies"][frequency]
    m, h = int(fcfg["season_length"]), int(fcfg["horizon"])
    union = set(samples.loc[samples["frequency"] == frequency, "unique_id"])
    data = load_dataset_pair({"frequency": frequency, **fcfg}, data_dir, load_manifest(manifest_path))
    data = data[data["unique_id"].isin(union)][["unique_id", "source_dataset", "t", "y"]]
    by_id = {uid: g for uid, g in data.groupby("unique_id", sort=True)}
    cutoffs = pd.read_parquet(prepare / "cutoffs.parquet")
    cutoffs = cutoffs[cutoffs["frequency"] == frequency]

    root = run_dir / "evaluate" / shard
    store = TaskStore(root, asdict(prov))
    failed = []
    with Heartbeat(root / "heartbeat.json", "evaluate", shard) as beat:
        for i, task in enumerate(tasks):
            beat.update(task=task.task_id, done=i, total=len(tasks), failed=len(failed))
            if store.is_done(task.task_id):
                continue
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
                attempts = []
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
                store.clear_failure(task.task_id)
            rows.insert(0, "task_id", task.task_id)
            sha = write_parquet(store.output(task.task_id), rows)
            store.complete(task.task_id, sha, {"rows": int(len(rows)), "failed_rows": int((rows["status"] != "trained").sum())})
        beat.update(task=None, done=len(tasks), failed=len(failed))
    if failed:
        raise StageError(f"{len(failed)} global tasks failed twice: {failed}")
    return {"shard": shard, "tasks": len(tasks)}
