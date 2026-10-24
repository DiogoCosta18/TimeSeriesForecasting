"""Revision runs E1, E1b and E3, without tuning.

- E1: STL-AC with periodic STL and with a seasonal smoother of span 25, for the nine global models (the run's
  frozen configurations of the T, S and R targets) on every feature sample, and for ARIMA and ETS (season length
  1 on T and R, m on S); cohort scope, main seed, both frequencies, the three windows. Each row also keeps the
  component forecasts and variant (ii) (T and R modelled, S continued by its last cycle; engine.ac_components).
- E1b: STL-AC periodic for the seven stochastic global models on the evolving-seasonality sample with the two
  extra seeds.
- E3: ARIMA and ETS on the log scale (Direct, STL-SN and STL-AC, with the run's STL and with periodic STL;
  season length 1 on the targets without seasonality, ETS with additive components only), on every series.

Everything lives in RUN/sensitivity_s6/; the stages are those of src.stages.sensitivity:

    freeze    no tuning: the run's frozen configurations, with the provenance of the freezing stage
    evaluate  --group cpu (ML; statistical chunks; --workers N) or gpu (neural, transformer); --shard S
    check     completeness (G5), commit and pins (G4), failed rows (G1), finite metrics (G9) and the controls

Controls: XGBoost under STL-AC with the run's STL on the evolving-seasonality sample (both frequencies),
bit-identical to the run's rows; ETS under STL-AC with season length 1 and the run's STL (one chunk per
frequency), bit-identical to S3's rows.

    python -m src.stages.sensitivity6 STAGE --config C --run RUN [--data-dir D] [--group G] [--workers N] [--shard S]
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.frozen import DEFAULT_MANIFEST
from src.data.schemas import FEATURE_NAMES
from src.forecast.registry import GLOBAL_MODELS, STRATEGY_TARGETS, config_key
from src.stages import sensitivity as s1
from src.stages import sensitivity3 as s3
from src.stages import sensitivity5 as s5
from src.stages.io import StageError, load_bundle, read_json, utc_now, write_json
from src.stages.tasks import FREQUENCIES, SEED_CHECK_FEATURE, STATISTICAL_CHUNK, STOCHASTIC_MODELS
from src.validation.gates import load_merged

log = logging.getLogger(__name__)

ROOT = "sensitivity_s6"
FROZEN_FORMAT = "rerun-s6-configs/1"
E1_VARIANTS = ["periodic", "s25"]
STATISTICAL = ["ARIMA", "ETS"]
E3_CELLS = [("direct", "elog"), ("stl_sn", "elog"), ("stl_sn", "elog_periodic"), ("stl_ac", "elog"), ("stl_ac", "elog_periodic")]
CONTROL_PREFIX = "s6ctrl"
EV = SEED_CHECK_FEATURE
EXTRA = ["t_hat", "s_hat", "r_hat", "s_sn", "yhat_ii", "relnaive_capped_ii", "relnaive_ii", "mae_ii"]


def study(config: dict) -> s1.Study:
    """S6 for a configuration: E1b's tasks depend on the extra seeds."""
    main_seed, seeds = int(config["random_seed"]), [int(x) for x in config["seed_check_seeds"]]

    def eval_tasks(bucket_summary: pd.DataFrame, samples: pd.DataFrame) -> list[dict]:
        features = [f for f in FEATURE_NAMES if f in set(bucket_summary["feature_name"])]
        tasks = []

        def glob(prefix, feature, frequency, model, variant, seed):
            tasks.append({"task_id": f"{prefix}|{feature}|{frequency}|{model}|stl_ac|{variant}|seed{seed}", "kind": "global",
                          "group": s1.group_of(model), "frequency": frequency, "feature_name": feature, "model": model,
                          "strategy": "stl_ac", "variant": variant, "seed": seed})

        def stat(prefix, model, frequency, strategy, variant, chunk):
            tasks.append({"task_id": f"{prefix}|{model}|{frequency}|{strategy}|{variant}|chunk{chunk:03d}", "kind": "statistical",
                          "group": "cpu", "frequency": frequency, "chunk": chunk, "model": model, "strategy": strategy,
                          "variant": variant})

        for frequency in FREQUENCIES:
            n_chunks = math.ceil(samples.loc[samples["frequency"] == frequency, "unique_id"].nunique() / STATISTICAL_CHUNK)
            for feature in features:
                for model in GLOBAL_MODELS:
                    for variant in E1_VARIANTS:
                        glob("s6e1", feature, frequency, model, variant, main_seed)
            for model in STOCHASTIC_MODELS:
                for seed in seeds:
                    glob("s6e1b", EV, frequency, model, "periodic", seed)
            for model in STATISTICAL:
                for chunk in range(n_chunks):
                    for variant in E1_VARIANTS:
                        stat("s6e1stat", model, frequency, "stl_ac", f"{variant}_p1", chunk)
                    for strategy, variant in E3_CELLS:
                        stat("s6e3", model, frequency, strategy, variant, chunk)
            glob(CONTROL_PREFIX, EV, frequency, "XGBoost", "default", main_seed)
            stat(CONTROL_PREFIX, "ETS", frequency, "stl_ac", "p1", 0)
        return tasks

    return s1.Study("s6", ROOT, lambda: [], eval_tasks, FROZEN_FORMAT, s5.run_entries, entries=entries)


def entries(task: dict, frozen: dict) -> dict:
    """The run's frozen configurations of the task's targets (T, S and R for STL-AC)."""
    return {key: frozen["entries"][key] for key in
            (config_key(task["model"], task["frequency"], target) for target in STRATEGY_TARGETS[task["strategy"]])}


def run_check(run_dir: Path, config: dict, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    """The checks of S6's rows, written to RUN/sensitivity_s6/checks.json; the analyses read S6 only if they pass."""
    run_dir = Path(run_dir)
    S = study(config)
    bundle = load_bundle(run_dir, config, manifest_path)
    frozen = s1.load_frozen(run_dir, S)
    prepare = run_dir / "prepare"
    samples = pd.read_parquet(prepare / "samples.parquet")
    tasks = S.eval_tasks(pd.read_parquet(prepare / "bucket_summary.parquet"), samples)
    columns = [*s1.ANALYSIS_COLUMNS, "relnaive_capped_ii", "relnaive_ii", "mae_ii"]
    rows, facts = s1._s1_rows(run_dir, tasks, ROOT, columns)
    record, run_rows, _ = load_merged(run_dir, ["model", "strategy", "scope", "feature_name", "frequency", "unique_id",
                                                 "window", "seed", "forecast_hash"])
    checks = {"G5_complete": {"expected": len(tasks), "missing": facts["missing"], "changed": facts["changed"],
                              "passed": not facts["missing"] and not facts["changed"]},
              "G4_commit_and_pins": {"commits": facts["commits"], "pinned": facts["pins"], "configs": facts["configs"],
                                     "frozen_configs": frozen["configs_sha256"],
                                     "bundle_matches_run": bundle["bundle_sha256"] == record["provenance"]["bundle_sha256"]}}
    g4 = checks["G4_commit_and_pins"]
    g4["passed"] = (len(facts["commits"]) == 1 and facts["pins"] == [True] and facts["configs"] == [frozen["configs_sha256"]]
                    and g4["bundle_matches_run"])
    trained = rows[rows["status"] == "trained"] if len(rows) else rows
    failed = rows[rows["status"] != "trained"] if len(rows) else rows
    checks["G1_trained_rows"] = {"rows": int(len(rows)), "failed_rows": int(len(failed)),
                                 "failed_series": sorted(set(failed["unique_id"])) if len(failed) else [],
                                 "listed_task_failures": facts["listed_failures"], "passed": not facts["listed_failures"]}
    metrics = ["relnaive_capped", "relnaive", "mase", "smape", "mae", "pocid"]
    nonfinite = int((~np.isfinite(trained[metrics].to_numpy(dtype=float))).any(axis=1).sum()) if len(trained) else 0
    ac = trained[(trained["strategy"] == "stl_ac") & ~trained["task_id"].astype(str).str.startswith(CONTROL_PREFIX)]
    nonfinite_ii = int((~np.isfinite(ac[["relnaive_capped_ii", "mae_ii"]].to_numpy(dtype=float))).any(axis=1).sum()) if len(ac) else 0
    checks["G9_finite"] = {"non_finite_rows": nonfinite, "non_finite_variant_ii": nonfinite_ii,
                           "passed": nonfinite == 0 and nonfinite_ii == 0}
    if len(rows):
        ctrl = rows[rows["task_id"].astype(str).str.startswith(CONTROL_PREFIX)]
        keys = ["frequency", "unique_id", "window"]
        xgb = ctrl[ctrl["model"] == "XGBoost"]
        ref = run_rows[(run_rows["model"] == "XGBoost") & (run_rows["strategy"] == "stl_ac") & (run_rows["scope"] == "cohort")
                       & (run_rows["feature_name"] == EV) & (run_rows["seed"] == int(config["random_seed"]))]
        j = xgb[keys + ["forecast_hash"]].merge(ref[keys + ["forecast_hash"]], on=keys, how="left", suffixes=("", "_run"),
                                                validate="one_to_one")
        same = j["forecast_hash"] == j["forecast_hash_run"]
        checks["control_xgboost_ac_reproduces_run"] = {"rows": int(len(j)), "identical": int(same.sum()),
                                                       "passed": len(j) > 0 and bool(same.all())}
        s3_tasks = [t for t in s3.eval_tasks(pd.read_parquet(prepare / "bucket_summary.parquet"), samples)
                    if t["task_id"].startswith("s3stat|ETS|") and t["strategy"] == "stl_ac" and t["chunk"] == 0]
        s3_rows, _ = s1._s1_rows(run_dir, s3_tasks, s3.ROOT, s1.ANALYSIS_COLUMNS)
        ets = ctrl[ctrl["model"] == "ETS"]
        j = ets[keys + ["forecast_hash"]].merge(s3_rows[keys + ["forecast_hash"]], on=keys, how="left", suffixes=("", "_s3"),
                                                validate="one_to_one")
        same = j["forecast_hash"] == j["forecast_hash_s3"]
        checks["control_ets_ac_p1_reproduces_s3"] = {"rows": int(len(j)), "identical": int(same.sum()),
                                                     "passed": len(j) > 0 and bool(same.all())}
    passed = all(c["passed"] for c in checks.values())
    write_json(run_dir / ROOT / "checks.json", {"passed": passed, "checks": checks, "created_at_utc": utc_now()})
    if not passed:
        raise StageError(f"S6 checks failed: {sorted(k for k, v in checks.items() if not v['passed'])}")
    return {"passed": passed, "rows": int(len(rows))}


def main(argv: list[str] | None = None) -> int:
    from src.utils import read_yaml

    parser = argparse.ArgumentParser(prog="python -m src.stages.sensitivity6", description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["freeze", "evaluate", "check"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--group", choices=["cpu", "gpu"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = read_yaml(args.config)
    S = study(config)
    if args.stage == "evaluate" and (args.group is None or args.data_dir is None):
        parser.error("evaluate needs --group and --data-dir")
    if args.stage == "freeze":
        print(s1.run_freeze(args.run, study=S, config=config))
    elif args.stage == "evaluate":
        print(s1.run_evaluate(args.run, config, args.data_dir, args.group, DEFAULT_MANIFEST, workers=args.workers, study=S,
                              shard=args.shard))
    else:
        print(run_check(args.run, config, DEFAULT_MANIFEST))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
