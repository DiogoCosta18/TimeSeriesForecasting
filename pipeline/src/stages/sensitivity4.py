"""Sensitivity analysis S4, the STL configuration for all twelve models (protocol change log v2.12 (f)).

STL-SN with (P) periodic STL as in S1 and (W) a seasonal smoother of span 25 and degree 1, robust;
ARIMA and ETS with season length 1 on T + R (the main specification, v2.12 (d)), the global models
with the run's frozen configurations of the non-seasonal target (no tuning); cohort scope, main
seed, every feature sample, both frequencies and the three windows of the run. Everything lives in
RUN/sensitivity_s4/; the stages are those of src.stages.sensitivity:

    tune      the tuning check: XGBoost, NHITS and PatchTST on the periodic non-seasonal target (6 studies)
    freeze    the 6 studies and the run's frozen configurations of the non-seasonal target
    evaluate  --group cpu (ML; statistical chunks; --workers N) or gpu (neural, transformer); --shard S
    analyse   the checks (G1, G4, G5, G6, G9 for S4) and the controls, then the tests against Direct and
              against the default STL-SN (the run's for the global models, S3's for ARIMA and ETS), per
              model and variant and per family, Wilcoxon with Holm; the tuning check; R2 and the
              gradients of R3 per variant (src.analysis.review; --strength: E7's table)

Controls: ETS under periodic STL with season length m on every series (with ``--reference-s1 RUN2``,
bit-identical to S1's ETS rows); the four ML models under the run's STL on the evolving-seasonality
sample (bit-identical to the run's rows); NHITS and PatchTST under the run's STL on the monthly
evolving-seasonality sample (median per-series absolute difference from the run within the median
seed spread of A15).

    python -m src.stages.sensitivity4 STAGE --config C --run RUN [--data-dir D] [--group G] [--workers N]
                                      [--shard S] [--reference-s1 RUN2] [--strength E7_TABLE]

S4 adds no confirmatory hypothesis. The reading is fixed in v2.12 (f).
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.data import ROW_COLUMNS, build_instances, per_series, strategy_contrast
from src.analysis.hypotheses import one_sample
from src.data.frozen import DEFAULT_MANIFEST
from src.data.schemas import FEATURE_NAMES
from src.forecast import tuning
from src.forecast.registry import FAMILY, GLOBAL_MODELS, config_key, family
from src.stages import sensitivity as s1
from src.stages.io import StageError, code_commit, load_bundle, read_json, sha256_json, utc_now, write_json
from src.stages.tasks import FREQUENCIES, STATISTICAL_CHUNK
from src.validation.gates import load_merged

log = logging.getLogger(__name__)

ROOT = "sensitivity_s4"
VARIANTS = ["periodic", "s25"]
STATISTICAL = ["ARIMA", "ETS"]
TUNED_MODELS = ["XGBoost", "NHITS", "PatchTST"]
ML_CONTROL = ["Ridge", "LinearRegression", "XGBoost", "RandomForest"]
NN_CONTROL = ["NHITS", "PatchTST"]
CONTROL_FEATURE = "feature_evolving_seasonality"
FROZEN_FORMAT = "rerun-s4-configs/1"
RUN_ENTRY = "run"
CONTROL_PREFIX = "s4ctrl"


def tune_tasks() -> list[dict]:
    return [{"task_id": f"s4tune|{model}|{frequency}|periodic", "model": model, "frequency": frequency,
             "variant": "periodic", "group": s1.group_of(model)} for model in TUNED_MODELS for frequency in FREQUENCIES]


def eval_tasks(bucket_summary: pd.DataFrame, samples: pd.DataFrame) -> list[dict]:
    features = [f for f in FEATURE_NAMES if f in set(bucket_summary["feature_name"])]
    tasks = []

    def glob(prefix, feature, frequency, model, variant, entry):
        tasks.append({"task_id": f"{prefix}|{feature}|{frequency}|{model}|{variant}|{entry}", "kind": "global",
                      "group": s1.group_of(model), "frequency": frequency, "feature_name": feature, "model": model,
                      "strategy": "stl_sn", "variant": variant, "entry": entry})

    for frequency in FREQUENCIES:
        n_chunks = math.ceil(samples.loc[samples["frequency"] == frequency, "unique_id"].nunique() / STATISTICAL_CHUNK)
        for feature in features:
            for model in GLOBAL_MODELS:
                for variant in VARIANTS:
                    glob("s4eval", feature, frequency, model, variant, RUN_ENTRY)
            for model in TUNED_MODELS:
                glob("s4eval", feature, frequency, model, "periodic", "tuned")
        for model in STATISTICAL:
            for variant in VARIANTS:
                for chunk in range(n_chunks):
                    tasks.append({"task_id": f"s4stat|{model}|{frequency}|{variant}_p1|chunk{chunk:03d}", "kind": "statistical",
                                  "group": "cpu", "frequency": frequency, "chunk": chunk, "model": model,
                                  "strategy": "stl_sn", "variant": f"{variant}_p1"})
        for chunk in range(n_chunks):            # control: S1's ETS under periodic STL, season length m
            tasks.append({"task_id": f"{CONTROL_PREFIX}|ETS|{frequency}|periodic|chunk{chunk:03d}", "kind": "statistical",
                          "group": "cpu", "frequency": frequency, "chunk": chunk, "model": "ETS", "strategy": "stl_sn",
                          "variant": "periodic"})
        if CONTROL_FEATURE in features:          # controls: the run's STL for ML (both frequencies) and two neural models
            for model in ML_CONTROL:
                glob(CONTROL_PREFIX, CONTROL_FEATURE, frequency, model, "default", RUN_ENTRY)
            if frequency == "monthly":
                for model in NN_CONTROL:
                    glob(CONTROL_PREFIX, CONTROL_FEATURE, frequency, model, "default", RUN_ENTRY)
    return tasks


def entry_name(model: str, frequency: str, entry: str) -> str:
    return s1.entry_key(model, frequency, "periodic" if entry == "tuned" else RUN_ENTRY)


def run_entries(run_dir: Path) -> dict:
    """The run's frozen configurations of the non-seasonal target for every global model (hash-checked)."""
    record = tuning.load_frozen_configs(Path(run_dir) / "configs_frozen.json")
    return {s1.entry_key(model, frequency, RUN_ENTRY): {**record["entries"][config_key(model, frequency, "nonseasonal")],
                                                       "taken_from_run_configs": record["configs_sha256"]}
            for model in GLOBAL_MODELS for frequency in FREQUENCIES}


def entries(task: dict, frozen: dict) -> dict:
    return {config_key(task["model"], task["frequency"], "nonseasonal"):
            frozen["entries"][entry_name(task["model"], task["frequency"], task["entry"])]}


S4 = s1.Study("s4", ROOT, tune_tasks, eval_tasks, FROZEN_FORMAT, run_entries, entries=entries)


# --- analyse ---------------------------------------------------------------------------------------------

KEYS = ["feature_name", "frequency", "model", "unique_id", "window"]
LABELS = ["periodic", "s25", "periodic_tuned"]                    # the arms of S4 in the analysis


def _is_control(rows: pd.DataFrame) -> pd.Series:
    return rows["task_id"].astype(str).str.startswith(CONTROL_PREFIX)


def _label(rows: pd.DataFrame) -> pd.Series:
    """The analysis arm of a row: its STL variant without the season-length suffix, ``periodic_tuned``
    for the tuning check."""
    v = rows["stl_variant"].astype(str).str.replace("_p1", "", regex=False)
    return v.where(~rows["task_id"].astype(str).str.endswith("|tuned"), "periodic_tuned")


def models_of(label: str) -> list[str]:
    return TUNED_MODELS if label == "periodic_tuned" else [*GLOBAL_MODELS, *STATISTICAL]


def ml_control(ctrl: pd.DataFrame, run_rows: pd.DataFrame) -> dict:
    a = ctrl[ctrl["family"] == "ml"]
    b = run_rows[(run_rows["strategy"] == "stl_sn") & run_rows["model"].isin(ML_CONTROL)]
    j = a[KEYS + ["forecast_hash"]].merge(b[KEYS + ["forecast_hash"]], on=KEYS, how="left", suffixes=("", "_run"),
                                           validate="one_to_one")
    same = j["forecast_hash"] == j["forecast_hash_run"]
    return {"rows": int(len(j)), "identical": int(same.sum()), "passed": len(j) > 0 and bool(same.all())}


def nn_control(ctrl: pd.DataFrame, run_rows: pd.DataFrame, spread: pd.DataFrame | None) -> dict:
    """Per series, the mean RelNaive(10) over the windows under the control and in the run; the median
    absolute difference must not exceed A15's median per-series range over the seeds (same scale).
    ``spread`` None: the run has no seed check (a pilot), so the bound does not exist and the control
    is reported without a verdict."""
    a = ctrl[ctrl["model"].isin(NN_CONTROL)]
    b = run_rows[(run_rows["strategy"] == "stl_sn") & run_rows["model"].isin(NN_CONTROL)
                 & run_rows["feature_name"].isin(set(a["feature_name"])) & run_rows["frequency"].isin(set(a["frequency"]))]
    keys = ["model", "unique_id"]
    mean_a = a.groupby(keys)["relnaive_capped"].mean().rename("control")
    mean_b = b.groupby(keys)["relnaive_capped"].mean().rename("run")
    j = pd.concat([mean_a, mean_b], axis=1).reset_index()
    unpaired = int(j[["control", "run"]].isna().any(axis=1).sum())
    out, ok = {}, len(a) > 0 and unpaired == 0
    for model in NN_CONTROL:
        g = j[j["model"] == model]
        diff = float((g["control"] - g["run"]).abs().median()) if len(g) else float("nan")
        out[model] = {"n_series": int(len(g)), "median_abs_diff": diff}
        if spread is not None:
            row = spread[(spread["model"] == model) & (spread["strategy"] == "stl_sn")]
            bound = float(row["median_range"].iloc[0]) if len(row) else float("nan")
            out[model]["seed_median_range"] = bound
            ok = ok and len(g) > 0 and diff <= bound
    if spread is None:
        return {"models": out, "unpaired_series": unpaired, "passed": True,
                "note": "the run has no seed check, so A15's bound does not exist (pilot): no verdict"}
    return {"models": out, "unpaired_series": unpaired, "passed": bool(ok)}


def s1_control(ctrl: pd.DataFrame, reference: Path | None) -> dict:
    if reference is None:
        return {"reference": None, "passed": True, "note": "no reference given; checked separately if at all"}
    prepare = Path(reference) / "prepare"
    samples = pd.read_parquet(prepare / "samples.parquet")
    tasks = [t for t in s1.eval_tasks(pd.read_parquet(prepare / "bucket_summary.parquet"), samples)
             if t["kind"] == "statistical" and t["variant"] == "periodic"]
    ref, facts = s1._s1_rows(reference, tasks, s1.ROOT)
    if facts["missing"] or facts["changed"]:
        return {"reference": str(reference), "passed": False, "missing": facts["missing"], "changed": facts["changed"]}
    keys = ["model", "unique_id", "window"]
    a = ctrl[ctrl["family"] == "statistical"][keys + ["forecast_hash"]]
    j = a.merge(ref[keys + ["forecast_hash"]], on=keys, how="outer", suffixes=("", "_s1"), validate="one_to_one",
                indicator=True)
    same = (j["_merge"] == "both") & (j["forecast_hash"] == j["forecast_hash_s1"])
    return {"reference": str(reference), "rows": int(len(j)), "identical": int(same.sum()),
            "only_s4": int((j["_merge"] == "left_only").sum()), "only_s1": int((j["_merge"] == "right_only").sum()),
            "passed": bool(same.all())}


def contrasts(inst: pd.DataFrame, first: str, labels: list[str]) -> pd.DataFrame:
    """Instance-level RelNaive(first) - RelNaive(STL-SN arm) for each arm, on the arm's models only."""
    parts = []
    for label in labels:
        d = strategy_contrast(inst, first, f"stl_sn_{label}")
        parts.append(d[d["model"].astype(str).isin(models_of(label))].assign(variant=label))
    return pd.concat(parts, ignore_index=True)


def run_analyse(run_dir: Path, config: dict, strength: Path | None, manifest_path: Path = DEFAULT_MANIFEST,
                reference_s1: Path | None = None, data_dir: Path | None = None) -> dict:
    """``strength``: E7's table of seasonal strength (R2's strong series and the gradients); without it,
    the strength is computed as E7 computes it, from the frozen copy in ``data_dir``."""
    from src.analysis import review
    from src.stages import sensitivity3 as s3

    run_dir = Path(run_dir)
    out = run_dir / ROOT / "analysis"
    if out.exists() and any(out.iterdir()):
        raise StageError(f"{out} is not empty; S4 outputs are never overwritten")
    s3_checks = run_dir / s3.ROOT / "analysis" / "checks.json"
    if not s3_checks.exists() or not read_json(s3_checks)["passed"]:
        raise StageError("S4 compares ARIMA and ETS with S3's STL-SN; S3 must be analysed with its checks passed first")
    commit = code_commit()
    bundle = load_bundle(run_dir, config, manifest_path)
    frozen = s1.load_frozen(run_dir, S4)
    prepare = run_dir / "prepare"
    samples, buckets = pd.read_parquet(prepare / "samples.parquet"), pd.read_parquet(prepare / "buckets.parquet")
    bucket_summary = pd.read_parquet(prepare / "bucket_summary.parquet")
    tasks = eval_tasks(bucket_summary, samples)
    s4_rows, facts = s1._s1_rows(run_dir, tasks, ROOT)
    s3_tasks = [t for t in s3.eval_tasks(bucket_summary, samples) if t["strategy"] == "stl_sn" and t["variant"] == s3.VARIANT]
    s3_rows, s3_facts = s1._s1_rows(run_dir, s3_tasks, s3.ROOT)
    record, rows, failed = load_merged(run_dir, ROW_COLUMNS + ["forecast_hash"])
    main_seed = int(config["random_seed"])
    models = [*GLOBAL_MODELS, *STATISTICAL]
    keep = rows["model"].isin(models) & (rows["scope"] == "cohort") & rows["strategy"].isin(["direct", "stl_sn"])
    run_rows = rows[keep & (rows["seed"].isna() | (rows["seed"] == main_seed))]

    s3_bad = s3_facts["missing"] + s3_facts["changed"]
    checks = {"G5_complete": {"expected": len(tasks), "missing": facts["missing"], "changed": facts["changed"],
                              "s3_missing_or_changed": s3_bad},
              "G4_commit_and_pins": {"commits": facts["commits"], "pinned": facts["pins"], "configs": facts["configs"],
                                     "frozen_configs": frozen["configs_sha256"],
                                     "bundle_matches_run": bundle["bundle_sha256"] == record["provenance"]["bundle_sha256"]}}
    checks["G5_complete"]["passed"] = not facts["missing"] and not facts["changed"] and not s3_bad
    checks["G4_commit_and_pins"]["passed"] = (len(facts["commits"]) == 1 and facts["pins"] == [True]
                                              and facts["configs"] == [frozen["configs_sha256"]]
                                              and checks["G4_commit_and_pins"]["bundle_matches_run"])
    trained = s4_rows[s4_rows["status"] == "trained"] if len(s4_rows) else s4_rows
    metrics = ["relnaive_capped", "relnaive", "mase", "smape", "mae", "pocid"]
    nonfinite = int((~np.isfinite(trained[metrics].to_numpy(dtype=float))).any(axis=1).sum()) if len(trained) else 0
    checks["G1_trained_rows"] = {"rows": int(len(s4_rows)), "failed_rows": int(len(s4_rows) - len(trained)),
                                 "listed_task_failures": facts["listed_failures"], "passed": not facts["listed_failures"]}
    checks["G9_finite"] = {"non_finite_rows": nonfinite, "passed": nonfinite == 0}
    if len(s4_rows):
        ctrl = s4_rows[_is_control(s4_rows)]
        spread = pd.read_csv(run_dir / "analysis" / "tables" / "a15_seed_spread.csv") if config["seed_check_seeds"] else None
        checks["control_ml_reproduces_run"] = ml_control(ctrl, run_rows)
        checks["control_nn_within_seed_spread"] = nn_control(ctrl, run_rows, spread)
        checks["control_ets_periodic_matches_s1"] = s1_control(ctrl, reference_s1)

    arms = s4_rows[~_is_control(s4_rows)] if len(s4_rows) else s4_rows
    arms = arms.assign(strategy="stl_sn_" + _label(arms)) if len(arms) else arms
    default = pd.concat([run_rows[(run_rows["strategy"] == "stl_sn") & (run_rows["family"] != "statistical")],
                         s3_rows], ignore_index=True).assign(strategy="stl_sn_default")
    direct = run_rows[run_rows["strategy"] == "direct"]
    both = pd.concat([direct[ROW_COLUMNS], default[ROW_COLUMNS], arms[ROW_COLUMNS]], ignore_index=True)
    inst = build_instances(both, failed.iloc[0:0], buckets, main_seed)
    inst = inst[inst["main_seed"]]
    vs_direct = contrasts(inst, "direct", ["default", *LABELS])
    vs_default = contrasts(inst, "stl_sn_default", LABELS)
    arm_inst = inst[inst["strategy"].astype(str).isin([f"stl_sn_{v}" for v in LABELS])]
    missing_arm = int(arm_inst["relnaive_capped"].isna().sum())
    unpaired = int(vs_direct[vs_direct["variant"] != "default"]["delta"].isna().sum()) - missing_arm
    unpaired_default = int(vs_default["delta"].isna().sum()) - missing_arm
    expected = {(m, v) for v in ["default", *LABELS] for m in models_of(v)}
    present = set(vs_direct.dropna(subset=["delta"]).groupby(["model", "variant"], observed=True).size().index)
    checks["G6_pairing"] = {"s4_instances": int(arm_inst["relnaive_capped"].notna().sum()), "unpaired_with_direct": unpaired,
                            "unpaired_with_default": unpaired_default,
                            "missing_contrasts": sorted(map(list, expected - present)),
                            "passed": unpaired == 0 and unpaired_default == 0 and expected <= present}
    passed = all(c["passed"] for c in checks.values())
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "checks.json", {"passed": passed, "checks": checks, "created_at_utc": utc_now()})
    if not passed:
        raise StageError(f"S4 checks failed: {sorted(k for k, v in checks.items() if not v['passed'])}")

    s0 = main_seed
    by, fam_by = ["model", "variant"], ["family", "variant"]
    arms_direct = vs_direct[vs_direct["variant"].isin(["periodic", "s25"])]
    arms_default = vs_default[vs_default["variant"].isin(["periodic", "s25"])]
    tuned = strategy_contrast(inst, "stl_sn_periodic", "stl_sn_periodic_tuned")
    tuned = tuned[tuned["model"].astype(str).isin(TUNED_MODELS)]
    tests = {
        "s4_vs_direct": one_sample(per_series(arms_direct, by), by, s0, "S4-vs-direct"),          # > 0: the arm beats Direct
        "s4_vs_default": one_sample(per_series(arms_default, by), by, s0, "S4-vs-default"),       # > 0: the arm beats the default
        "s4_family_vs_direct": one_sample(per_series(arms_direct, fam_by), fam_by, s0, "S4-family-vs-direct"),
        "s4_family_vs_default": one_sample(per_series(arms_default, fam_by), fam_by, s0, "S4-family-vs-default"),
        "s4_tuning_check": one_sample(per_series(tuned, ["model"]), ["model"], s0, "S4-tuning-check"),  # > 0: tuning helps
    }
    features = pd.read_parquet(prepare / "features.parquet")
    if strength is not None:
        strength_table = pd.read_csv(strength)
    elif data_dir is not None:
        from src.analysis.exploratory2 import strength_table as e7_strength
        strength_table = e7_strength(config, data_dir, features, samples[["unique_id", "frequency"]].drop_duplicates(),
                                     manifest_path)
    else:
        raise StageError("S4's analysis needs E7's strength table (--strength) or the frozen copy (--data-dir)")
    x, _ = review.cells(features, bucket_summary, strength_table)
    grads = vs_direct[vs_direct["variant"].isin(["default", "periodic", "s25"])]
    pooled = per_series(grads, ["frequency", "variant"], family_balanced=True).astype({"unique_id": str, "frequency": str})
    fam = per_series(grads, ["frequency", "variant", "family"]).astype({"unique_id": str, "frequency": str, "family": str})
    terc, corr = review.gradients(pooled, x, s0, ["variant"], "S4-R3")
    means = review.series_model_means(inst)
    r2 = pd.concat([review.r2_best_direct(means, x, s0, f"stl_sn_{v}", "R2" if v == "default" else f"R2[{v}]").assign(variant=v)
                    for v in ("default", "periodic", "s25")], ignore_index=True)
    tables = {**tests, "s4_gradients": terc, "s4_correlations": corr,
              "s4_family_terciles": review.family_terciles(fam, x, ["variant"]), "s4_r2": r2}
    (out / "tables").mkdir()
    for name, df in tables.items():
        df.to_csv(out / "tables" / f"{name}.csv", index=False)
    summary = {"status": "sensitivity analysis S4 (protocol change log v2.12 (f)); not a confirmatory hypothesis",
               "analysis_code_commit": commit, "created_at_utc": utc_now(),
               "strength_table": str(strength) if strength is not None else "computed as E7 (no table given)",
               "merge_result_sha256": record["result_sha256"], "s4_configs_sha256": frozen["configs_sha256"],
               "default_stl_sn": "the run's for the global models, S3's (season length 1) for ARIMA and ETS",
               "families": sorted(set(FAMILY[m] for m in models)),
               "tables": {name: sha256_json(df.to_dict(orient="list")) for name, df in tables.items()}}
    write_json(out / "s4.json", summary)
    return {"tests": sum(len(t) for t in tests.values()) + len(r2)}


def main(argv: list[str] | None = None) -> int:
    from src.utils import read_yaml

    parser = argparse.ArgumentParser(prog="python -m src.stages.sensitivity4", description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["tune", "freeze", "evaluate", "analyse"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--group", choices=["cpu", "gpu"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard")
    parser.add_argument("--reference-s1", type=Path)
    parser.add_argument("--strength", type=Path, help="E7's e7_strength.csv (analyse)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = read_yaml(args.config)
    if args.stage in ("tune", "evaluate") and (args.group is None or args.data_dir is None):
        parser.error(f"{args.stage} needs --group and --data-dir")
    if args.stage == "analyse" and args.strength is None and args.data_dir is None:
        parser.error("analyse needs --strength (E7's table) or --data-dir")
    if args.stage == "tune":
        print(s1.run_tune(args.run, config, args.data_dir, args.group, DEFAULT_MANIFEST, study=S4))
    elif args.stage == "freeze":
        print(s1.run_freeze(args.run, study=S4, config=config))
    elif args.stage == "evaluate":
        print(s1.run_evaluate(args.run, config, args.data_dir, args.group, DEFAULT_MANIFEST, workers=args.workers, study=S4,
                              shard=args.shard))
    else:
        print(run_analyse(args.run, config, args.strength, DEFAULT_MANIFEST, args.reference_s1, args.data_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
