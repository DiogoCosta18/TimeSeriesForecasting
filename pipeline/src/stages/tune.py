"""Stages R2 (tune, per shard) and R2b (freeze the configurations) of protocol Section 9.1.

R2 runs the shard's studies on the tuning set of the prepare bundle; every study is
checked to end at the first cutoff (gate G7) and its frozen entry is written with a
completion record. A failed study is retried once (as global tasks are, D16); a second
failure is listed and the stage ends with an error. R2b collects the 90 entries,
checks that every shard used the same bundle, data, code and environment, and writes
configs_frozen.json with its content hash.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

import optuna
import pandas as pd

from src.data.frozen import DEFAULT_MANIFEST, load_manifest, sha256_file
from src.data.load_m_datasets import load_dataset_pair
from src.forecast import tuning
from src.stages.io import Heartbeat, StageError, TaskStore, load_bundle, read_json, stage_provenance, write_json
from src.stages.tasks import tuning_tasks

log = logging.getLogger(__name__)


def _drop_partial_study(storage_path: Path, study_name: str) -> None:
    """A study archived by an attempt that crashed before its completion record is removed and redone."""
    if Path(storage_path).exists():
        storage = f"sqlite:///{Path(storage_path).resolve()}"
        if study_name in optuna.get_all_study_names(storage):
            optuna.delete_study(study_name=study_name, storage=storage)


def run_tune(run_dir: Path, config: dict, data_dir: Path, shard: str, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    run_dir = Path(run_dir)
    provenance = stage_provenance(load_bundle(run_dir, config, manifest_path), manifest_path)
    root = run_dir / "tune" / shard
    store = TaskStore(root, provenance)
    tasks = [t for t in tuning_tasks() if t.shard == shard]
    if not tasks:
        raise StageError(f"no tuning tasks in shard {shard!r}")
    frequency = tasks[0].frequency
    fcfg = config["frequencies"][frequency]
    m, h = int(fcfg["season_length"]), int(fcfg["horizon"])
    data = load_dataset_pair({"frequency": frequency, **fcfg}, data_dir, load_manifest(manifest_path))
    prepare = run_dir / "prepare"
    tuning_set = pd.read_parquet(prepare / "tuning_set.parquet")
    cutoffs = pd.read_parquet(prepare / "cutoffs.parquet")
    first_cutoff = cutoffs[(cutoffs["frequency"] == frequency) & (cutoffs["window"] == 0)].set_index("unique_id")["train_end_idx"]
    storage = root / "studies.sqlite"
    failed = []
    with Heartbeat(root / "heartbeat.json", "tune", shard) as beat:
        for i, task in enumerate(tasks):
            beat.update(task=task.task_id, done=i, total=len(tasks), failed=len(failed))
            if store.is_done(task.task_id, ".json"):
                continue
            frame = tuning.tuning_frame(data, tuning_set, frequency, task.target, m)
            ends = frame.groupby("unique_id")["ds"].max()
            if not (ends == first_cutoff.loc[ends.index]).all():  # G7
                raise StageError(f"{task.task_id}: tuning data do not end at the first cutoff")
            attempts = []
            for _ in range(2):
                try:
                    _drop_partial_study(storage, tuning.config_key(task.model, frequency, task.target).replace("|", "__"))
                    entry = tuning.tune_study(task.model, frequency, task.target, frame, h, m, config["tuning"],
                                              int(config["random_seed"]), storage)
                    break
                except Exception as exc:  # D16: retried once, then listed
                    attempts.append(f"{type(exc).__name__}: {exc}")
                    log.error("%s failed:\n%s", task.task_id, traceback.format_exc())
            else:
                store.record_failure(task.task_id, attempts)
                failed.append(task.task_id)
                continue
            store.clear_failure(task.task_id)
            entry["g7_tuning_end_equals_first_cutoff"] = True
            write_json(store.output(task.task_id, ".json"), entry)
            store.complete(task.task_id, sha256_file(store.output(task.task_id, ".json")), suffix=".json")
        beat.update(task=None, done=len(tasks), failed=len(failed))
    if failed:
        raise StageError(f"{len(failed)} studies failed twice: {failed}")
    return {"shard": shard, "studies": len(tasks)}


def run_freeze(run_dir: Path, config: dict) -> str:
    """Collect every study's entry into configs_frozen.json; return its content hash."""
    run_dir = Path(run_dir)
    target = run_dir / "configs_frozen.json"
    if target.exists():
        raise StageError(f"{target} exists; frozen configurations are never overwritten")
    entries, provenances = {}, set()
    for task in tuning_tasks():
        record_path = run_dir / "tune" / task.shard / "tasks" / f"{task.task_id.replace('|', '__')}.done.json"
        if not record_path.exists():
            raise StageError(f"study {task.task_id} has not completed")
        record = read_json(record_path)
        output = record_path.with_name(record["file"])
        if sha256_file(output) != record["sha256"]:
            raise StageError(f"{task.task_id}: entry changed since it was recorded")
        entry = read_json(output)
        entries[tuning.config_key(task.model, task.frequency, task.target)] = entry
        provenances.add(tuple(sorted(record["provenance"].items())))
    if len(provenances) != 1:
        raise StageError(f"studies were run with different provenance: {sorted(provenances)}")
    provenance = dict(provenances.pop())
    return tuning.write_frozen_configs(target, entries, config["tuning"], provenance)
