"""A small synthetic frozen copy and configuration for the stage tests.

Test-only: the pipeline itself never generates data. Each source holds eight regular
series, one too short for L (SHORT) and one with a constant feature history (FLAT).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from helpers_frozen import write_frozen
from src.data.frozen import COLUMNS

COMMIT = "a" * 40
CONFIG = {
    "random_seed": 7,
    "seed_check_seeds": [8, 9],
    "frequencies": {
        "monthly": {"season_length": 12, "horizon": 18, "m3_group": "Monthly", "m4_group": "Monthly"},
        "quarterly": {"season_length": 4, "horizon": 8, "m3_group": "Quarterly", "m4_group": "Quarterly"},
    },
    "validation": {"n_windows": 3},
    "sampling": {"n_per_source": 6, "n_strata": 3},
    "buckets": {"min_share": 0.2},
    "tuning_set": {"n_per_source": 6},
    "tuning": {"num_samples": 1, "early_stopping": {"patience": 1, "check_steps": 5}},
}
SPEC = {"Monthly": (12, 120, 100), "Quarterly": (4, 50, 40)}  # m, regular length, short length
SOURCES = ["M3_Monthly", "M4_Monthly", "M3_Quarterly", "M4_Quarterly"]


def canonical(key: str, seed: int) -> pd.DataFrame:
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


def synthetic_frozen_copy(root: Path) -> tuple[Path, Path]:
    """Write the synthetic frozen copy under ``root`` (the data directory); return (root, manifest_path)."""
    root.mkdir(parents=True, exist_ok=True)
    _, manifest_path = write_frozen(root, {key: canonical(key, i) for i, key in enumerate(SOURCES)})
    return root, manifest_path


def fake_fits(patch) -> None:
    """Replace the model fits by cheap deterministic forecasts (a per-model level); the
    stages, the engine, STL, metrics and provenance stay real. ``patch`` is a MonkeyPatch.
    Every fit reports one second, so STL-AC / STL-SN fit time is exactly 3 (G10)."""
    from src.forecast import models
    from src.forecast.registry import BACKEND, MODELS, SOURCE, family
    from src.forecast.result import ForecastResult

    def level(y, model):
        return float(np.mean(np.asarray(y, dtype=float)[-4:])) + 0.01 * MODELS.index(model)

    def forecast_statistical(model, y_train, h, season_length, timings=None):
        if timings is not None:
            timings.update(fit_seconds=1.0, predict_seconds=0.1)
        return ForecastResult(np.full(h, level(y_train, model)), SOURCE["statistical"], "statsforecast", "test", h)

    def fit_predict_global(model, params, train, h, season_length, seed, *, trained_steps=None, config_key=None, timings=None):
        if timings is not None:
            timings.update(fit_seconds=1.0, predict_seconds=0.1)
        pool = 0.001 * train["unique_id"].nunique()  # the pool matters, so specialisation has an effect
        return {uid: ForecastResult(np.full(h, level(g["y"], model) + pool), SOURCE[family(model)], BACKEND[family(model)], "test", h,
                                    config_key=config_key, seed=seed)
                for uid, g in train.groupby("unique_id", sort=True)}

    patch.setattr(models, "forecast_statistical", forecast_statistical)
    patch.setattr(models, "fit_predict_global", fit_predict_global)


def complete_studies(run: Path, provenance: dict) -> None:
    """Completed tuning outputs for all 90 studies (as stage R2 writes them)."""
    import optuna

    from src.stages import io
    from src.stages.io import safe_name
    from src.stages.tasks import tuning_tasks

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for task in tuning_tasks():
        store = io.TaskStore(run / "tune" / task.shard, provenance)
        entry = {"model": task.model, "frequency": task.frequency, "target": task.target, "params": {},
                 "best_validation_mase": 1.0, "g7_tuning_end_equals_first_cutoff": True}
        if task.family != "ml":
            entry["trained_steps"] = 1
        io.write_json(store.output(task.task_id, ".json"), entry)
        store.complete(task.task_id, io.sha256_file(store.output(task.task_id, ".json")), suffix=".json")
        study = optuna.create_study(storage=f"sqlite:///{(run / 'tune' / task.shard / 'studies.sqlite').resolve()}",
                                    study_name=safe_name(f"{task.model}|{task.frequency}|{task.target}"), direction="minimize")
        for value in (1.2, 1.0):
            study.add_trial(optuna.trial.create_trial(params={}, distributions={}, value=value, user_attrs={"duration_seconds": 1.0}))


def evaluated_run(root: Path, config: dict) -> tuple[Path, Path, Path]:
    """prepare -> (synthetic) tune -> freeze -> evaluate every shard, with fake fits.

    Returns (run_dir, data_dir, manifest_path)."""
    import pytest

    from src.stages import io
    from src.stages.evaluate import run_evaluate
    from src.stages.prepare import run_prepare
    from src.stages.tasks import evaluation_tasks
    from src.stages.tune import run_freeze

    data_dir, manifest_path = synthetic_frozen_copy(root / "data")
    run = root / "run"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RERUN_CODE_COMMIT", COMMIT)
        fake_fits(patch)
        bundle = run_prepare(config, run, data_dir, COMMIT, manifest_path=manifest_path)
        complete_studies(run, io.stage_provenance(bundle, manifest_path))
        run_freeze(run, config)
        prepare = run / "prepare"
        tasks = evaluation_tasks(pd.read_parquet(prepare / "bucket_summary.parquet"), pd.read_parquet(prepare / "samples.parquet"),
                                 int(config["random_seed"]), list(config["seed_check_seeds"]))
        for shard in sorted({t.shard for t in tasks}):
            run_evaluate(run, config, data_dir, shard, manifest_path)
    return run, data_dir, manifest_path
