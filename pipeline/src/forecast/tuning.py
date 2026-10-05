"""Tuning and frozen configurations (protocol Section 4.3, D12, D13, D17; test U12).

One study per global model x frequency x target (raw, T+R, T, S, R) on the tuning set:
every series truncated at its first cutoff; the last h points of each series form a
single validation block (Auto classes: NeuralForecast ``val_size=h``; AutoMLForecast
``n_windows=1, h=h``, whose splits are per series). Library Auto classes with Optuna's
TPE sampler (seeded per study), ``num_samples`` trials, objective mean validation MASE
with seasonality m. Early stopping is used only here; the frozen configuration stores
the sampled parameters and, for neural models, the number of steps the best
configuration actually trained (read from its refit, which uses the same validation).

Targets: for STL-SN and STL-AC, STL is fitted once on each tuning series (positions
1 .. first cutoff) and the component series are the targets; nothing after the first
cutoff is used.

Every trial is archived in SQLite (JSON-safe attributes only). The frozen-configuration
file is hashed over its content (not its creation time); evaluation refuses a file whose
hash does not match.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from src.forecast import spaces
from src.forecast.models import accelerator
from src.forecast.registry import GLOBAL_MODELS, TARGETS, config_key, family
from src.forecast.targets import component_targets

FROZEN_FORMAT = "rerun-frozen-configs/1"
FREQUENCY_CODE = {"monthly": 12, "quarterly": 4}


class TuningError(RuntimeError):
    """A study did not produce a usable configuration."""


def _sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def study_seed(base_seed: int, model: str, frequency: str, target: str) -> int:
    """TPE seed of one study, derived from the base seed (one stream per study)."""
    ss = np.random.SeedSequence([int(base_seed), GLOBAL_MODELS.index(model), FREQUENCY_CODE[frequency], TARGETS.index(target)])
    return int(ss.generate_state(1)[0])


def tuning_frame(data: pd.DataFrame, tuning_set: pd.DataFrame, frequency: str, target: str, season_length: int) -> pd.DataFrame:
    """unique_id, ds (= position t), y (= target) of every tuning series of ``frequency``,
    truncated at its first cutoff (``tuning_validation_end_t`` of the prepare bundle)."""
    rows = tuning_set[tuning_set["frequency"] == frequency]
    if rows.empty:
        raise TuningError(f"no tuning series for {frequency}")
    end = dict(zip(rows["unique_id"], rows["tuning_validation_end_t"].astype(int)))
    sub = data[data["unique_id"].isin(end)].sort_values(["unique_id", "t"], kind="mergesort")
    parts = []
    for uid, g in sub.groupby("unique_id", sort=True):
        y = g["y"].to_numpy(dtype=float)[: end[uid]]
        if len(y) != end[uid]:
            raise TuningError(f"{uid}: {len(y)} points, expected {end[uid]} before the first cutoff")
        values = y if target == "raw" else component_targets(y, season_length)[target]
        parts.append(pd.DataFrame({"unique_id": uid, "ds": np.arange(1, len(y) + 1, dtype="int64"), "y": values}))
    if len(parts) != len(end):
        raise TuningError(f"{len(end) - len(parts)} tuning series are missing from the data")
    return pd.concat(parts, ignore_index=True)


def _mase_loss(season_length: int):
    from utilsforecast.losses import mase

    def loss(df, train_df):  # AutoMLForecast calls loss(valid, train_df=train)
        return float(mase(df, models=["model"], seasonality=season_length, train_df=train_df)["model"].mean())

    return loss


def tune_ml(model: str, frame: pd.DataFrame, h: int, season_length: int, fit_seed: int, tpe_seed: int,
            num_samples: int) -> tuple[optuna.Study, dict]:
    from mlforecast.auto import AutoMLForecast, AutoModel

    def init_config(trial):
        choice = trial.suggest_categorical("target_transform", spaces.TARGET_TRANSFORMS)
        return {"lags": spaces.ml_lags(season_length),
                "target_transforms": spaces.ml_target_transforms({"target_transform": choice}, season_length)}

    def model_config(trial):  # estimator parameters built by the same function evaluation uses
        return spaces.ml_estimator(model, spaces.sample_ml(model, trial), fit_seed).get_params()

    template = spaces.ml_estimator(model, _any_ml_params(model), fit_seed)
    auto = AutoMLForecast(models={model: AutoModel(template, model_config)}, freq=1, init_config=init_config)
    # No pruning: every trial is evaluated on the validation block and archived with its MASE
    # (Section 5.1). Optuna's default median pruner would end trials after their only window
    # once 5 trials exist, leaving them without a value (neuralforecast reports nothing to prune).
    auto.fit(frame, n_windows=1, h=h, num_samples=num_samples, loss=_mase_loss(season_length),
             study_kwargs={"sampler": optuna.samplers.TPESampler(seed=tpe_seed), "pruner": optuna.pruners.NopPruner()})
    study = auto.results_[model]
    return study, dict(study.best_trial.params)


def _any_ml_params(model: str) -> dict:
    trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)).ask()
    return spaces.sample_ml(model, trial)


def tune_neural(model: str, frame: pd.DataFrame, h: int, season_length: int, fit_seed: int, tpe_seed: int,
                num_samples: int, early_stopping: dict) -> tuple[optuna.Study, dict, int]:
    from neuralforecast import NeuralForecast
    from neuralforecast.common._base_auto import MockTrial
    from neuralforecast.losses.pytorch import MAE, MASE

    device = accelerator()

    def kwargs_for(params: dict) -> dict:
        kw = spaces.neural_kwargs(model, params, h, fit_seed, max_steps=params["max_steps"], accelerator=device,
                                  early_stopping=early_stopping)
        kw.pop("h")  # the Auto class sets h itself
        return kw

    def config(trial):
        if isinstance(trial, MockTrial):  # neuralforecast reads only the keys of this first call
            probe = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)).ask()
            return {k: None for k in kwargs_for(spaces.sample_neural(model, probe, season_length))}
        return kwargs_for(spaces.sample_neural(model, trial, season_length))

    auto = spaces.neural_auto_class(model)(
        h=h, loss=MAE(), valid_loss=MASE(seasonality=season_length), config=config,
        search_alg=optuna.samplers.TPESampler(seed=tpe_seed), num_samples=num_samples, backend="optuna",
    )
    nf = NeuralForecast(models=[auto], freq=1)
    nf.fit(frame, val_size=h)
    fitted = nf.models[0]
    study = fitted.results
    steps = fitted.model.train_trajectories
    trained_steps = int(steps[-1][0]) + 1 if steps else 0
    if trained_steps < 1 or trained_steps != len(steps):
        raise TuningError(f"{model}: cannot read the trained steps of the best configuration ({len(steps)} entries)")
    return study, dict(study.best_trial.params), trained_steps


def archived_trials(study: optuna.Study, study_name: str) -> list[optuna.trial.FrozenTrial]:
    """Every trial as it is archived: parameters, distributions, validation MASE, start and end
    times and duration (Section 5.1); the libraries' own attributes are not JSON-safe and are
    left out (which also lets a worker process hand the trials back)."""
    bad = [t.number for t in study.trials if t.state != optuna.trial.TrialState.COMPLETE]
    if bad:
        raise TuningError(f"{study_name}: trials {bad} did not complete")
    return [
        optuna.trial.FrozenTrial(
            number=t.number, state=t.state, value=t.value, values=None,
            datetime_start=t.datetime_start, datetime_complete=t.datetime_complete,
            params=t.params, distributions=t.distributions,
            user_attrs={"duration_seconds": (t.datetime_complete - t.datetime_start).total_seconds()},
            system_attrs={}, intermediate_values={}, trial_id=t._trial_id,
        )
        for t in study.trials
    ]


def archive_study(trials: list[optuna.trial.FrozenTrial], storage_path: Path, study_name: str) -> None:
    """Copy the archived trials of one study into SQLite."""
    storage = f"sqlite:///{Path(storage_path).resolve()}"
    archive = optuna.create_study(storage=storage, study_name=study_name, direction="minimize")
    archive.add_trials(trials)


def study_name(model: str, frequency: str, target: str) -> str:
    return config_key(model, frequency, target).replace("|", "__")


def run_study(model: str, frequency: str, target: str, frame: pd.DataFrame, h: int, season_length: int,
              settings: dict, base_seed: int) -> tuple[dict, list[optuna.trial.FrozenTrial]]:
    """Run one study; return its frozen-configuration entry (without the archive reference)
    and its trials to archive. Nothing is written, so studies can run in worker processes."""
    key = config_key(model, frequency, target)
    tpe_seed = study_seed(base_seed, model, frequency, target)
    num_samples = int(settings["num_samples"])
    entry = {"model": model, "frequency": frequency, "target": target, "fit_seed": int(base_seed),
             "tpe_seed": tpe_seed, "num_samples": num_samples, "n_tuning_series": int(frame["unique_id"].nunique())}
    if family(model) == "ml":
        study, params = tune_ml(model, frame, h, season_length, int(base_seed), tpe_seed, num_samples)
    else:
        study, params, trained_steps = tune_neural(model, frame, h, season_length, int(base_seed), tpe_seed,
                                                   num_samples, settings["early_stopping"])
        entry["trained_steps"] = trained_steps
    if len(study.trials) != num_samples:
        raise TuningError(f"{key}: {len(study.trials)} trials, expected {num_samples}")
    trials = archived_trials(study, study_name(model, frequency, target))
    entry.update(params=params, best_validation_mase=float(study.best_value))
    return entry, trials


def tune_study(model: str, frequency: str, target: str, frame: pd.DataFrame, h: int, season_length: int,
               settings: dict, base_seed: int, storage_path: Path) -> dict:
    """Run one study, archive its trials and return its frozen-configuration entry."""
    entry, trials = run_study(model, frequency, target, frame, h, season_length, settings, base_seed)
    name = study_name(model, frequency, target)
    archive_study(trials, storage_path, name)
    entry["study"] = f"{Path(storage_path).name}:{name}"
    return entry


def write_frozen_configs(path: Path, entries: dict, settings: dict, provenance: dict) -> str:
    """Write configs_frozen.json; return its content hash (also stored in the file)."""
    body = {"format": FROZEN_FORMAT, "settings": settings, "provenance": provenance, "entries": entries}
    digest = _sha256_json(body)
    record = {**body, "configs_sha256": digest,
              "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return digest


def load_frozen_configs(path: Path) -> dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("format") != FROZEN_FORMAT:
        raise TuningError(f"{path}: not a frozen-configuration file")
    body = {k: record[k] for k in ("format", "settings", "provenance", "entries")}
    if _sha256_json(body) != record.get("configs_sha256"):
        raise TuningError(f"{path}: content does not match its hash")
    for key, entry in record["entries"].items():
        if key != config_key(entry["model"], entry["frequency"], entry["target"]):
            raise TuningError(f"{path}: entry {key} is inconsistent")
        if family(entry["model"]) != "ml" and int(entry.get("trained_steps", 0)) < 1:
            raise TuningError(f"{path}: {key} has no trained steps")
    return record
