"""Revision run E2: the global models re-tuned for STL-SN with periodic STL.

One study per model and frequency on the periodic non-seasonal target (T + R of periodic STL on the training
part of the tuning series), with the run's tuning settings in everything else (tuning set, validation block,
search spaces, 40 TPE trials with the run's sampler seed, objective, step rule; src.forecast.tuning, as S1 and
S4). S4 already re-tuned XGBoost, NHITS and PatchTST in this way and evaluated them on every series; E2 adds the
other six global models (12 studies) and evaluates them with their new configurations under STL-SN periodic on
every feature sample, cohort scope, main seed, both frequencies and the three windows. Everything lives in
RUN/sensitivity_s7/; the stages are those of src.stages.sensitivity:

    tune      --group cpu (LinearRegression, Ridge, RandomForest) or gpu (NLinear, LSTM, TFT)
    freeze    the 12 entries -> configs_frozen.json
    evaluate  --group cpu (--workers N) or gpu; --shard S

    python -m src.stages.sensitivity7 STAGE --config C --run RUN [--data-dir D] [--group G] [--workers N] [--shard S]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data.frozen import DEFAULT_MANIFEST
from src.data.schemas import FEATURE_NAMES
from src.forecast.registry import GLOBAL_MODELS
from src.stages import sensitivity as s1
from src.stages import sensitivity4 as s4
from src.stages.tasks import FREQUENCIES

ROOT = "sensitivity_s7"
FROZEN_FORMAT = "rerun-s7-configs/1"
MODELS = [m for m in GLOBAL_MODELS if m not in s4.TUNED_MODELS]


def tune_tasks() -> list[dict]:
    return [{"task_id": f"s7tune|{model}|{frequency}|periodic", "model": model, "frequency": frequency,
             "variant": "periodic", "group": s1.group_of(model)} for model in MODELS for frequency in FREQUENCIES]


def eval_tasks(bucket_summary: pd.DataFrame, samples: pd.DataFrame) -> list[dict]:
    features = [f for f in FEATURE_NAMES if f in set(bucket_summary["feature_name"])]
    return [{"task_id": f"s7eval|{feature}|{frequency}|{model}|periodic|tuned", "kind": "global", "group": s1.group_of(model),
             "frequency": frequency, "feature_name": feature, "model": model, "strategy": "stl_sn", "variant": "periodic"}
            for frequency in FREQUENCIES for feature in features for model in MODELS]


S7 = s1.Study("s7", ROOT, tune_tasks, eval_tasks, FROZEN_FORMAT)


def main(argv: list[str] | None = None) -> int:
    from src.utils import read_yaml

    parser = argparse.ArgumentParser(prog="python -m src.stages.sensitivity7", description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["tune", "freeze", "evaluate"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--group", choices=["cpu", "gpu"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = read_yaml(args.config)
    if args.stage in ("tune", "evaluate") and (args.group is None or args.data_dir is None):
        parser.error(f"{args.stage} needs --group and --data-dir")
    if args.stage == "tune":
        print(s1.run_tune(args.run, config, args.data_dir, args.group, DEFAULT_MANIFEST, study=S7))
    elif args.stage == "freeze":
        print(s1.run_freeze(args.run, study=S7))
    else:
        print(s1.run_evaluate(args.run, config, args.data_dir, args.group, DEFAULT_MANIFEST, workers=args.workers, study=S7,
                              shard=args.shard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
