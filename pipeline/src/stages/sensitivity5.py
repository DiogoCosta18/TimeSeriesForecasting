"""Sensitivity analysis S5, homogeneity or pool size (protocol change log v2.12 (h)).

For every feature sample and frequency, the series of the sample are split at random (numpy seed
sequence of the main seed and the CRC-32 of "S5|feature|frequency") into groups of the sizes of its
populated buckets. Every tercile task of the grid gets a twin trained on the random group of the same
size (scopes ``rlow``, ``rmedium``, ``rhigh``), with the same frozen configurations, windows and seed;
each series is forecast by the model of its own group and keeps its own bucket as the pairing key. A
second arm retrains the tercile and random tasks of NLinear, NHITS, LSTM, TFT and PatchTST with a third
of each target's frozen training steps (rounded, at least one; scopes ``low_s3``, ``rlow_s3``, ...).
Everything lives in RUN/sensitivity_s5/; the stages are those of src.stages.sensitivity:

    partitions  (no forecasts) the random groups and their checks: sizes, every series in one group, and
                the grouping feature's values of each random group against the rest of the sample
                (Kolmogorov-Smirnov, Holm); run on the bundle before any machine is rented
    freeze      no tuning: the run's frozen configurations, with the provenance of the freezing stage
    evaluate    --group cpu (ML; --workers N) or gpu (neural, transformer); --shard S
    analyse     the checks (G1, G4, G5, G6, G9 for S5), the partition checks and the control, then per
                series Delta_FT = cohort - tercile, Delta_rand = cohort - random and the contrast
                random - tercile (> 0: grouping by the feature helps beyond the pool size), per family and
                strategy (Holm over nine), per model and strategy, linear against non-linear as in H6;
                the same for the second arm

Control: the four ML models' tercile tasks of the evolving-seasonality sample, recomputed through this
study's pools and entries, must be bit-identical to the run's rows.

    python -m src.stages.sensitivity5 STAGE --config C --run RUN [--data-dir D] [--group G] [--workers N] [--shard S]

S5 adds no confirmatory hypothesis. The reading is fixed in v2.12 (h).
"""
from __future__ import annotations

import argparse
import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from src.analysis import stats
from src.analysis.data import ROW_COLUMNS, _wide, build_instances, per_series
from src.analysis.hypotheses import _finish, _test, one_sample
from src.data.frozen import DEFAULT_MANIFEST
from src.data.schemas import FEATURE_NAMES
from src.forecast import tuning
from src.forecast.registry import GLOBAL_MODELS, LINEARITY, STRATEGIES, STRATEGY_TARGETS, config_key
from src.stages import sensitivity as s1
from src.stages.io import StageError, code_commit, load_bundle, sha256_json, utc_now, write_json
from src.stages.tasks import BUCKET_SCOPE, FREQUENCIES
from src.validation.gates import load_merged

log = logging.getLogger(__name__)

ROOT = "sensitivity_s5"
FROZEN_FORMAT = "rerun-s5-configs/1"
REDUCED = ["NLinear", "NHITS", "LSTM", "TFT", "PatchTST"]          # the second arm
ML_CONTROL = ["Ridge", "LinearRegression", "XGBoost", "RandomForest"]
CONTROL_FEATURE = "feature_evolving_seasonality"
CONTROL_PREFIX = "s5ctrl"
THIRD = "_s3"
BUCKETS = ["Low", "Medium", "High"]
SCOPE_BUCKET = {v: k for k, v in BUCKET_SCOPE.items()}           # low -> Low


def tune_tasks() -> list[dict]:
    return []


def populated(bucket_summary: pd.DataFrame, feature: str, frequency: str) -> list[str]:
    row = bucket_summary[(bucket_summary["frequency"] == frequency) & (bucket_summary["feature_name"] == feature)]
    if len(row) != 1:
        raise ValueError(f"bucket summary must hold one row for {feature} {frequency}")
    return row["populated_buckets"].iloc[0].split(",")


def eval_tasks(bucket_summary: pd.DataFrame, samples: pd.DataFrame) -> list[dict]:
    features = [f for f in FEATURE_NAMES if f in set(bucket_summary["feature_name"])]
    tasks = []

    def add(prefix, feature, frequency, strategy, model, scope):
        tasks.append({"task_id": f"{prefix}|{feature}|{frequency}|{strategy}|{model}|{scope}", "kind": "global",
                      "group": s1.group_of(model), "frequency": frequency, "feature_name": feature, "model": model,
                      "strategy": strategy, "scope": scope, "variant": "default"})

    for frequency in FREQUENCIES:
        for feature in features:
            for strategy in STRATEGIES:
                for model in GLOBAL_MODELS:
                    for bucket in populated(bucket_summary, feature, frequency):
                        scope = BUCKET_SCOPE[bucket]
                        add("s5eval", feature, frequency, strategy, model, f"r{scope}")
                        if model in REDUCED:
                            add("s5eval", feature, frequency, strategy, model, f"{scope}{THIRD}")
                            add("s5eval", feature, frequency, strategy, model, f"r{scope}{THIRD}")
                        if feature == CONTROL_FEATURE and model in ML_CONTROL:
                            add(CONTROL_PREFIX, feature, frequency, strategy, model, scope)
    return tasks


# --- pools and entries -----------------------------------------------------------------------------------

def sample_buckets(buckets: pd.DataFrame, samples: pd.DataFrame, feature: str, frequency: str) -> dict[str, str]:
    """Every series of a feature sample and its tercile bucket (checked against the sample)."""
    sample = buckets[(buckets["frequency"] == frequency) & (buckets["feature_name"] == feature)]
    bucket_of = dict(zip(sample["unique_id"], sample["feature_tercile_bucket"]))
    expected = set(samples[(samples["frequency"] == frequency) & (samples["feature_name"] == feature)]["unique_id"])
    if set(bucket_of) != expected:
        raise ValueError(f"{feature} {frequency}: buckets and sample disagree")
    return bucket_of


def partition(bucket_of: dict[str, str], feature: str, frequency: str, main_seed: int) -> dict[str, str]:
    """The random group (Low, Medium, High: the size of that bucket) of every series of the sample."""
    ids = sorted(bucket_of)
    sizes = pd.Series(list(bucket_of.values())).value_counts()
    rng = np.random.default_rng(np.random.SeedSequence([int(main_seed), zlib.crc32(f"S5|{feature}|{frequency}".encode())]))
    order = rng.permutation(len(ids))
    group, start = {}, 0
    for bucket in BUCKETS:
        n = int(sizes.get(bucket, 0))
        for i in order[start:start + n]:
            group[ids[i]] = bucket
        start += n
    return group


def members(task: dict, samples: pd.DataFrame, buckets: pd.DataFrame, main_seed: int) -> tuple[list[str], dict[str, str]]:
    bucket_of = sample_buckets(buckets, samples, task["feature_name"], task["frequency"])
    scope = task["scope"].removesuffix(THIRD)
    if scope.startswith("r"):
        group = partition(bucket_of, task["feature_name"], task["frequency"], main_seed)
        ids = sorted(u for u, g in group.items() if g == SCOPE_BUCKET[scope[1:]])
    else:
        ids = sorted(u for u, b in bucket_of.items() if b == SCOPE_BUCKET[scope])
    if not ids:
        raise ValueError(f"{task['task_id']}: empty training pool")
    return ids, {u: bucket_of[u] for u in ids}


def third(steps: int) -> int:
    return max(1, int(round(int(steps) / 3)))


def run_entries(run_dir: Path) -> dict:
    """All the run's frozen configurations, keyed as in the run (hash-checked)."""
    record = tuning.load_frozen_configs(Path(run_dir) / "configs_frozen.json")
    return {key: {**entry, "taken_from_run_configs": record["configs_sha256"]} for key, entry in record["entries"].items()}


def entries(task: dict, frozen: dict) -> dict:
    out = {}
    for target in STRATEGY_TARGETS[task["strategy"]]:
        key = config_key(task["model"], task["frequency"], target)
        entry = dict(frozen["entries"][key])
        if task["scope"].endswith(THIRD):
            entry["trained_steps"] = third(entry["trained_steps"])
        out[key] = entry
    return out


S5 = s1.Study("s5", ROOT, tune_tasks, eval_tasks, FROZEN_FORMAT, run_entries, members=members, entries=entries)


# --- partition checks ------------------------------------------------------------------------------------

def partition_table(samples: pd.DataFrame, buckets: pd.DataFrame, features: pd.DataFrame, bucket_summary: pd.DataFrame,
                    main_seed: int) -> pd.DataFrame:
    """One row per series of every feature sample: its bucket, its random group and its feature value."""
    names = [f for f in FEATURE_NAMES if f in set(bucket_summary["feature_name"])]
    value = features.set_index(["unique_id", "frequency"])
    rows = []
    for frequency in FREQUENCIES:
        for feature in names:
            bucket_of = sample_buckets(buckets, samples, feature, frequency)
            group = partition(bucket_of, feature, frequency, main_seed)
            for uid in sorted(bucket_of):
                rows.append({"feature_name": feature, "frequency": frequency, "unique_id": uid, "bucket": bucket_of[uid],
                             "random_group": group.get(uid), "value": float(value.loc[(uid, frequency), feature])})
    return pd.DataFrame(rows)


def partition_checks(table: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Sizes equal, every series in exactly one group, and KS of each random group's feature values
    against the rest of its sample (Holm over all groups); the check passes when no group differs."""
    sizes, ks = [], []
    for (feature, frequency), g in table.groupby(["feature_name", "frequency"], sort=True):
        a, b = g["bucket"].value_counts(), g["random_group"].value_counts()
        sizes.append(bool(a.sort_index().equals(b.sort_index())) and g["random_group"].notna().all())
        for group, h in g.groupby("random_group", sort=True):
            rest = g.loc[g["random_group"] != group, "value"]
            res = sps.ks_2samp(h["value"], rest)
            ks.append({"feature_name": feature, "frequency": frequency, "random_group": group, "n": len(h),
                       "ks_statistic": float(res.statistic), "p": float(res.pvalue)})
    ks = pd.DataFrame(ks)
    ks["p_holm"] = stats.holm(ks["p"].to_numpy())
    once = not table.duplicated(["feature_name", "frequency", "unique_id"]).any()
    check = {"sizes_equal_bucket_sizes": all(sizes), "every_series_once": bool(once),
             "ks_groups": int(len(ks)), "ks_min_p_holm": float(ks["p_holm"].min()),
             "passed": all(sizes) and once and bool((ks["p_holm"] >= 0.05).all())}
    return check, ks


def run_partitions(run_dir: Path, config: dict) -> dict:
    """The random groups and their checks, written to RUN/sensitivity_s5/partitions/ (no forecasts)."""
    run_dir = Path(run_dir)
    out = run_dir / ROOT / "partitions"
    if out.exists() and any(out.iterdir()):
        raise StageError(f"{out} is not empty; S5 outputs are never overwritten")
    commit = code_commit()                      # refuses an uncommitted tree before anything is written
    prepare = run_dir / "prepare"
    table = partition_table(pd.read_parquet(prepare / "samples.parquet"), pd.read_parquet(prepare / "buckets.parquet"),
                            pd.read_parquet(prepare / "features.parquet"), pd.read_parquet(prepare / "bucket_summary.parquet"),
                            int(config["random_seed"]))
    check, ks = partition_checks(table)
    out.mkdir(parents=True)
    table.to_csv(out / "partitions.csv", index=False)
    ks.to_csv(out / "ks.csv", index=False)
    write_json(out / "checks.json", {**check, "code_commit": commit, "created_at_utc": utc_now(),
                                     "partitions_sha256": sha256_json(table.to_dict(orient="list"))})
    if not check["passed"]:
        raise StageError(f"S5 partition checks failed: {check}; S5 is not run")
    return check


# --- analyse ---------------------------------------------------------------------------------------------

ARM_KEYS = ["feature_name", "frequency", "unique_id", "source_dataset", "window", "family", "model", "strategy", "bucket",
            "seed_key"]


def arm_of(scope: pd.Series) -> pd.Series:
    s = scope.astype(str)
    base = s.str.removesuffix(THIRD)
    arm = np.select([s == "cohort", base.str.startswith("r")], ["cohort", "random"], "tercile")
    return pd.Series(arm, index=scope.index) + np.where(s.str.endswith(THIRD), THIRD, "")


def deltas(inst: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """Per instance Delta_FT (cohort - tercile), Delta_rand (cohort - random) and the contrast (random -
    tercile), long format with a column ``quantity``; ``suffix`` "_s3" takes the second arm's pools."""
    wide = _wide(inst.assign(arm=arm_of(inst["scope"])), ARM_KEYS, "arm", "relnaive_capped")
    for arm in ("cohort", f"tercile{suffix}", f"random{suffix}"):
        if arm not in wide.columns:
            wide[arm] = np.nan
    q = {"delta_ft": wide["cohort"] - wide[f"tercile{suffix}"], "delta_rand": wide["cohort"] - wide[f"random{suffix}"],
         "contrast": wide[f"random{suffix}"] - wide[f"tercile{suffix}"]}
    keys = wide.index.to_frame(index=False)
    return pd.concat([keys.assign(quantity=name, delta=v.to_numpy()) for name, v in q.items()], ignore_index=True)


def linearity_test(d: pd.DataFrame, base_seed: int, family_id: str) -> pd.DataFrame:
    """As H6: non-linear minus linear per-series values, families weighted equally within each type;
    Holm over the three strategies of each quantity, as in H6."""
    d = d.assign(linearity=d["model"].astype(str).map(LINEARITY))
    parts = []
    for quantity, q in d.groupby("quantity", sort=True):
        out, fid = [], f"{family_id}[{quantity}]"
        ps = per_series(q, ["strategy", "linearity"], family_balanced=True)
        for strategy, g in ps.groupby("strategy", observed=True, sort=True):
            w = g.pivot(index="unique_id", columns="linearity", values="delta").dropna()
            if not {"linear", "nonlinear"} <= set(w.columns):
                continue
            x = (w["nonlinear"] - w["linear"]).to_numpy()
            out.append({"quantity": quantity, "strategy": strategy, "median_nonlinear": float(w["nonlinear"].median()),
                        "median_linear": float(w["linear"].median()), **_test(x, base_seed, f"{fid}|{strategy}")})
        parts.append(_finish(pd.DataFrame(out), fid))
    return pd.concat(parts, ignore_index=True)


def tables_for(d: pd.DataFrame, s0: int, tag: str) -> dict[str, pd.DataFrame]:
    out = {}
    for name, by in (("family", ["quantity", "family", "strategy"]), ("model", ["quantity", "model", "strategy"])):
        parts = [one_sample(per_series(q, by), by, s0, f"{tag}-{name}[{quantity}]")
                 for quantity, q in d.groupby("quantity", sort=True)]
        out[f"{tag.lower()}_by_{name}"] = pd.concat(parts, ignore_index=True)
    out[f"{tag.lower()}_linearity"] = linearity_test(d, s0, f"{tag}-linearity")
    return out


def run_analyse(run_dir: Path, config: dict, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    run_dir = Path(run_dir)
    out = run_dir / ROOT / "analysis"
    if out.exists() and any(out.iterdir()):
        raise StageError(f"{out} is not empty; S5 outputs are never overwritten")
    commit = code_commit()
    bundle = load_bundle(run_dir, config, manifest_path)
    frozen = s1.load_frozen(run_dir, S5)
    prepare = run_dir / "prepare"
    samples, buckets = pd.read_parquet(prepare / "samples.parquet"), pd.read_parquet(prepare / "buckets.parquet")
    bucket_summary = pd.read_parquet(prepare / "bucket_summary.parquet")
    main_seed = int(config["random_seed"])
    tasks = eval_tasks(bucket_summary, samples)
    s5_rows, facts = s1._s1_rows(run_dir, tasks, ROOT)
    record, rows, failed = load_merged(run_dir, ROW_COLUMNS + ["forecast_hash"])
    glob = rows[(rows["family"] != "statistical") & (rows["seed"] == main_seed)]

    checks = {"G5_complete": {"expected": len(tasks), "missing": facts["missing"], "changed": facts["changed"]},
              "G4_commit_and_pins": {"commits": facts["commits"], "pinned": facts["pins"], "configs": facts["configs"],
                                     "frozen_configs": frozen["configs_sha256"],
                                     "bundle_matches_run": bundle["bundle_sha256"] == record["provenance"]["bundle_sha256"]}}
    checks["G5_complete"]["passed"] = not facts["missing"] and not facts["changed"]
    checks["G4_commit_and_pins"]["passed"] = (len(facts["commits"]) == 1 and facts["pins"] == [True]
                                              and facts["configs"] == [frozen["configs_sha256"]]
                                              and checks["G4_commit_and_pins"]["bundle_matches_run"])
    trained = s5_rows[s5_rows["status"] == "trained"] if len(s5_rows) else s5_rows
    metrics = ["relnaive_capped", "relnaive", "mase", "smape", "mae", "pocid"]
    nonfinite = int((~np.isfinite(trained[metrics].to_numpy(dtype=float))).any(axis=1).sum()) if len(trained) else 0
    checks["G1_trained_rows"] = {"rows": int(len(s5_rows)), "failed_rows": int(len(s5_rows) - len(trained)),
                                 "listed_task_failures": facts["listed_failures"], "passed": not facts["listed_failures"]}
    checks["G9_finite"] = {"non_finite_rows": nonfinite, "passed": nonfinite == 0}
    table = partition_table(samples, buckets, pd.read_parquet(prepare / "features.parquet"), bucket_summary, main_seed)
    checks["partitions"], ks = partition_checks(table)
    is_ctrl = s5_rows["task_id"].astype(str).str.startswith(CONTROL_PREFIX) if len(s5_rows) else pd.Series(dtype=bool)
    if len(s5_rows):
        keys = ["feature_name", "frequency", "strategy", "model", "scope", "unique_id", "window"]
        ctrl = s5_rows[is_ctrl][keys + ["forecast_hash"]]
        j = ctrl.merge(glob[keys + ["forecast_hash"]], on=keys, how="left", suffixes=("", "_run"), validate="one_to_one")
        same = j["forecast_hash"] == j["forecast_hash_run"]
        checks["control_ml_tercile_reproduces_run"] = {"rows": int(len(j)), "identical": int(same.sum()),
                                                       "passed": len(j) > 0 and bool(same.all())}

    arms = s5_rows[~is_ctrl] if len(s5_rows) else s5_rows
    both = pd.concat([glob[ROW_COLUMNS], arms[ROW_COLUMNS]], ignore_index=True)
    inst = build_instances(both, failed.iloc[0:0], buckets, main_seed)
    inst = inst[inst["main_seed"]]
    full = deltas(inst)
    reduced = deltas(inst[inst["model"].astype(str).isin(REDUCED)], THIRD)
    tercile = full[full["quantity"] == "delta_ft"]
    has_tercile = tercile["delta"].notna()
    twin = full[full["quantity"] == "delta_rand"]["delta"].notna().to_numpy()
    r_terc = reduced[reduced["quantity"] == "delta_ft"]["delta"].notna().to_numpy()
    r_twin = reduced[reduced["quantity"] == "delta_rand"]["delta"].notna().to_numpy()
    checks["G6_pairing"] = {"tercile_instances": int(has_tercile.sum()),
                            "tercile_without_random_twin": int((has_tercile.to_numpy() & ~twin).sum()),
                            "random_without_tercile": int((~has_tercile.to_numpy() & twin).sum()),
                            "second_arm_tercile_without_twin": int((r_terc & ~r_twin).sum()),
                            "second_arm_random_without_tercile": int((~r_terc & r_twin).sum())}
    g6 = checks["G6_pairing"]
    g6["passed"] = g6["tercile_instances"] > 0 and not any(g6[k] for k in (
        "tercile_without_random_twin", "random_without_tercile", "second_arm_tercile_without_twin",
        "second_arm_random_without_tercile"))
    passed = all(c["passed"] for c in checks.values())
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "checks.json", {"passed": passed, "checks": checks, "created_at_utc": utc_now()})
    if not passed:
        raise StageError(f"S5 checks failed: {sorted(k for k, v in checks.items() if not v['passed'])}")

    s0 = main_seed
    tables = {"s5_ks": ks, **tables_for(full, s0, "S5"), **tables_for(reduced, s0, "S5S3")}
    (out / "tables").mkdir()
    for name, df in tables.items():
        df.to_csv(out / "tables" / f"{name}.csv", index=False)
    summary = {"status": "sensitivity analysis S5 (protocol change log v2.12 (h)); not a confirmatory hypothesis",
               "analysis_code_commit": commit, "created_at_utc": utc_now(),
               "merge_result_sha256": record["result_sha256"], "s5_configs_sha256": frozen["configs_sha256"],
               "partitions_sha256": sha256_json(table.to_dict(orient="list")),
               "tables": {name: sha256_json(df.to_dict(orient="list")) for name, df in tables.items()}}
    write_json(out / "s5.json", summary)
    return {"tests": sum(len(t) for k, t in tables.items() if k != "s5_ks")}


def main(argv: list[str] | None = None) -> int:
    from src.utils import read_yaml

    parser = argparse.ArgumentParser(prog="python -m src.stages.sensitivity5", description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["partitions", "freeze", "evaluate", "analyse"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--group", choices=["cpu", "gpu"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = read_yaml(args.config)
    if args.stage == "evaluate" and (args.group is None or args.data_dir is None):
        parser.error("evaluate needs --group and --data-dir")
    if args.stage == "partitions":
        print(run_partitions(args.run, config))
    elif args.stage == "freeze":
        print(s1.run_freeze(args.run, study=S5, config=config))
    elif args.stage == "evaluate":
        print(s1.run_evaluate(args.run, config, args.data_dir, args.group, DEFAULT_MANIFEST, workers=args.workers, study=S5,
                              shard=args.shard))
    else:
        print(run_analyse(args.run, config, DEFAULT_MANIFEST))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
