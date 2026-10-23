"""Seed check of S4 (protocol change log v2.15 (a)).

STL-SN with periodic STL (S4's (P)) for the seven stochastic global models on the evolving-seasonality
sample, both frequencies and the three windows, with the extra seeds of A15 and the run's frozen
configurations of the non-seasonal target. Everything lives in RUN/sensitivity_s4s/; the stages are those
of src.stages.sensitivity:

    freeze    no tuning: the run's frozen configurations, with the provenance of the freezing stage
    evaluate  --group cpu (RandomForest, XGBoost and the control; --workers N) or gpu (neural, transformer)
    analyse   the checks (G1, G4, G5, G6, G9) and the control, then, for each seed, the view of A15 (stochastic
              models from that seed, the others from the main seed) per family against Direct on the
              evolving-seasonality sample, Wilcoxon with Holm over the four families within each seed

Control: XGBoost under (P) with the main seed on the same sample, bit-identical to S4's rows.

    python -m src.stages.sensitivity4s STAGE --config C --run RUN [--data-dir D] [--group G] [--workers N]

The seeds come from the configuration (``random_seed``, ``seed_check_seeds``). No confirmatory hypothesis.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.data import ROW_COLUMNS, build_instances, per_series, strategy_contrast
from src.analysis.hypotheses import one_sample
from src.data.frozen import DEFAULT_MANIFEST
from src.forecast.registry import GLOBAL_MODELS
from src.stages import sensitivity as s1
from src.stages import sensitivity4 as s4
from src.stages.io import StageError, code_commit, load_bundle, read_json, sha256_json, utc_now, write_json
from src.stages.tasks import FREQUENCIES, SEED_CHECK_FEATURE, STOCHASTIC_MODELS
from src.validation.gates import load_merged

log = logging.getLogger(__name__)

ROOT = "sensitivity_s4s"
FROZEN_FORMAT = "rerun-s4s-configs/1"
CONTROL_PREFIX = "s4sctrl"
EV = SEED_CHECK_FEATURE
STATISTICAL = s4.STATISTICAL


def study(config: dict) -> s1.Study:
    """S4s for a configuration: its tasks depend on the main seed and the extra seeds."""
    main_seed, seeds = int(config["random_seed"]), [int(x) for x in config["seed_check_seeds"]]
    if not seeds:
        raise StageError("S4s needs the extra seeds of the seed check (seed_check_seeds)")

    def eval_tasks(bucket_summary: pd.DataFrame, samples: pd.DataFrame) -> list[dict]:
        if EV not in set(bucket_summary["feature_name"]):
            raise StageError(f"S4s needs the {EV} sample")
        tasks = []

        def add(prefix, frequency, model, seed):
            tasks.append({"task_id": f"{prefix}|{EV}|{frequency}|{model}|periodic|seed{seed}", "kind": "global",
                          "group": s1.group_of(model), "frequency": frequency, "feature_name": EV, "model": model,
                          "strategy": "stl_sn", "variant": "periodic", "entry": s4.RUN_ENTRY, "seed": seed})

        for frequency in FREQUENCIES:
            for model in STOCHASTIC_MODELS:
                for seed in seeds:
                    add("s4seval", frequency, model, seed)
            add(CONTROL_PREFIX, frequency, "XGBoost", main_seed)
        return tasks

    return s1.Study("s4s", ROOT, lambda: [], eval_tasks, FROZEN_FORMAT, s4.run_entries, entries=s4.entries)


def _view(seed: int, main_seed: int, s4s_rows: pd.DataFrame, s4_rows: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    """A15's view of one seed: the stochastic models from that seed, the others from the main seed."""
    stochastic = s4_rows["model"].isin(STOCHASTIC_MODELS)
    if seed == main_seed:
        periodic = s4_rows
    else:
        periodic = pd.concat([s4_rows[~stochastic], s4s_rows[s4s_rows["seed"] == seed]], ignore_index=True)
    d_stoch = direct["model"].isin(STOCHASTIC_MODELS)
    d = pd.concat([direct[~d_stoch & (direct["seed"].isna() | (direct["seed"] == main_seed))],
                   direct[d_stoch & (direct["seed"] == seed)]], ignore_index=True)
    return pd.concat([d[ROW_COLUMNS], periodic.assign(strategy="stl_sn_periodic")[ROW_COLUMNS]], ignore_index=True)


def run_analyse(run_dir: Path, config: dict, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    run_dir = Path(run_dir)
    S = study(config)
    out = run_dir / ROOT / "analysis"
    if out.exists() and any(out.iterdir()):
        raise StageError(f"{out} is not empty; S4s outputs are never overwritten")
    s4_checks = run_dir / s4.ROOT / "analysis" / "checks.json"
    if not s4_checks.exists() or not read_json(s4_checks)["passed"]:
        raise StageError("S4s extends S4; S4 must be analysed with its checks passed first")
    commit = code_commit()
    bundle = load_bundle(run_dir, config, manifest_path)
    frozen = s1.load_frozen(run_dir, S)
    prepare = run_dir / "prepare"
    samples, buckets = pd.read_parquet(prepare / "samples.parquet"), pd.read_parquet(prepare / "buckets.parquet")
    bucket_summary = pd.read_parquet(prepare / "bucket_summary.parquet")
    main_seed, seeds = int(config["random_seed"]), [int(x) for x in config["seed_check_seeds"]]
    tasks = S.eval_tasks(bucket_summary, samples)
    rows, facts = s1._s1_rows(run_dir, tasks, ROOT, s1.ANALYSIS_COLUMNS)
    s4_tasks = [t for t in s4.eval_tasks(bucket_summary, samples)
                if (t["task_id"].startswith("s4eval|") and t["feature_name"] == EV and t["variant"] == "periodic"
                    and t["entry"] == s4.RUN_ENTRY)
                or (t["task_id"].startswith("s4stat|") and t["variant"] == "periodic_p1")]
    s4_rows, s4_facts = s1._s1_rows(run_dir, s4_tasks, s4.ROOT, s1.ANALYSIS_COLUMNS)
    record, run_rows, failed = load_merged(run_dir, ROW_COLUMNS + ["forecast_hash"])
    models = [*GLOBAL_MODELS, *STATISTICAL]
    direct = run_rows[(run_rows["strategy"] == "direct") & (run_rows["scope"] == "cohort") & run_rows["model"].isin(models)
                      & ((run_rows["feature_name"] == EV) | (run_rows["family"] == "statistical"))]
    del run_rows

    s4_bad = s4_facts["missing"] + s4_facts["changed"]
    checks = {"G5_complete": {"expected": len(tasks), "missing": facts["missing"], "changed": facts["changed"],
                              "s4_missing_or_changed": s4_bad},
              "G4_commit_and_pins": {"commits": facts["commits"], "pinned": facts["pins"], "configs": facts["configs"],
                                     "frozen_configs": frozen["configs_sha256"],
                                     "bundle_matches_run": bundle["bundle_sha256"] == record["provenance"]["bundle_sha256"]}}
    checks["G5_complete"]["passed"] = not facts["missing"] and not facts["changed"] and not s4_bad
    checks["G4_commit_and_pins"]["passed"] = (len(facts["commits"]) == 1 and facts["pins"] == [True]
                                              and facts["configs"] == [frozen["configs_sha256"]]
                                              and checks["G4_commit_and_pins"]["bundle_matches_run"])
    trained = rows[rows["status"] == "trained"] if len(rows) else rows
    metrics = ["relnaive_capped", "relnaive", "mase", "smape", "mae", "pocid"]
    nonfinite = int((~np.isfinite(trained[metrics].to_numpy(dtype=float))).any(axis=1).sum()) if len(trained) else 0
    checks["G1_trained_rows"] = {"rows": int(len(rows)), "failed_rows": int(len(rows) - len(trained)),
                                 "listed_task_failures": facts["listed_failures"], "passed": not facts["listed_failures"]}
    checks["G9_finite"] = {"non_finite_rows": nonfinite, "passed": nonfinite == 0}
    is_ctrl = rows["task_id"].astype(str).str.startswith(CONTROL_PREFIX) if len(rows) else pd.Series(dtype=bool)
    if len(rows):
        keys = ["frequency", "model", "unique_id", "window"]
        ctrl = rows[is_ctrl][keys + ["forecast_hash"]]
        ref = s4_rows[(s4_rows["model"] == "XGBoost")][keys + ["forecast_hash"]]
        j = ctrl.merge(ref, on=keys, how="left", suffixes=("", "_s4"), validate="one_to_one")
        same = j["forecast_hash"] == j["forecast_hash_s4"]
        checks["control_xgboost_matches_s4"] = {"rows": int(len(j)), "identical": int(same.sum()),
                                                "passed": len(j) > 0 and bool(same.all())}

    arms = rows[~is_ctrl] if len(rows) else rows
    deltas, unpaired = {}, {}
    for seed in [main_seed, *seeds]:
        inst = build_instances(_view(seed, main_seed, arms, s4_rows, direct), failed.iloc[0:0], buckets, main_seed)
        inst = inst[inst["feature_name"].astype(str) == EV]
        d = strategy_contrast(inst, "direct", "stl_sn_periodic")
        arm = inst[inst["strategy"].astype(str) == "stl_sn_periodic"]
        unpaired[seed] = int(d["delta"].isna().sum() - arm["relnaive_capped"].isna().sum())
        present = set(d.dropna(subset=["delta"])["family"].astype(str))
        unpaired[seed] += 0 if present == {"statistical", "ml", "neural", "transformer"} else 10 ** 6
        deltas[seed] = d
    checks["G6_pairing"] = {"unpaired_by_seed": {str(k): v for k, v in unpaired.items()},
                            "passed": all(v == 0 for v in unpaired.values())}
    passed = all(c["passed"] for c in checks.values())
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "checks.json", {"passed": passed, "checks": checks, "created_at_utc": utc_now()})
    if not passed:
        raise StageError(f"S4s checks failed: {sorted(k for k, v in checks.items() if not v['passed'])}")

    tests = pd.concat([one_sample(per_series(d, ["family"]), ["family"], main_seed, f"S4s[{seed}]").assign(seed=seed)
                       for seed, d in deltas.items()], ignore_index=True)
    labels = tests.pivot(index="family", columns="seed", values="label")
    robust = {f: bool((labels.loc[f] == labels.loc[f, main_seed]).all()) for f in labels.index}
    (out / "tables").mkdir()
    tests.to_csv(out / "tables" / "s4s_seed_families.csv", index=False)
    summary = {"status": "seed check of S4 (protocol change log v2.15 (a)); not a confirmatory hypothesis",
               "analysis_code_commit": commit, "created_at_utc": utc_now(), "seeds": [main_seed, *seeds],
               "merge_result_sha256": record["result_sha256"], "s4s_configs_sha256": frozen["configs_sha256"],
               "label_kept_under_every_seed": robust, "reading": ("robust to the seed" if all(robust.values()) else
                                                                  "labels per seed; a family whose label changes reads as no change"),
               "tables": {"s4s_seed_families": sha256_json(tests.to_dict(orient="list"))}}
    write_json(out / "s4s.json", summary)
    return {"tests": len(tests), "robust": robust}


def main(argv: list[str] | None = None) -> int:
    from src.utils import read_yaml

    parser = argparse.ArgumentParser(prog="python -m src.stages.sensitivity4s", description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["freeze", "evaluate", "analyse"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--group", choices=["cpu", "gpu"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = read_yaml(args.config)
    S = study(config)
    if args.stage == "evaluate" and (args.group is None or args.data_dir is None):
        parser.error("evaluate needs --group and --data-dir")
    if args.stage == "freeze":
        print(s1.run_freeze(args.run, study=S, config=config))
    elif args.stage == "evaluate":
        print(s1.run_evaluate(args.run, config, args.data_dir, args.group, DEFAULT_MANIFEST, workers=args.workers, study=S))
    else:
        print(run_analyse(args.run, config, DEFAULT_MANIFEST))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
