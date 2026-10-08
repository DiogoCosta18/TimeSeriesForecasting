"""Sensitivity analysis S2, what in the STL configuration matters (protocol change log v2.4).

STL-SN with three variants that separate what periodic STL (S1) changes, against the run's
Direct forecasts and against the run's own STL-SN; cohort scope, main seed, every feature
sample, both frequencies and the three windows of the run:

    deg0   the run's STL with a seasonal smoother of degree 0 (span unchanged)   ETS, XGBoost, NHITS, PatchTST
    last3  the run's STL; the continuation is the mean of the last three cycles  ETS, XGBoost, NHITS, PatchTST
    stlf   the run's STL; the model is fitted without a seasonal component      ETS, ARIMA

Everything lives in RUN/sensitivity_s2/; the stages are those of src.stages.sensitivity:

    tune      one study per global model and frequency on the deg0 non-seasonal target (6 studies)
    freeze    the 6 deg0 entries and, for last3, the run's frozen non-seasonal entries (hash-checked)
    evaluate  --group gpu (NHITS, PatchTST) or cpu (XGBoost; ETS and ARIMA chunks; --workers N)
    analyse   the checks (as gates G1, G4, G5, G6, G9 for S2), then two families of 10 Wilcoxon tests
              with Holm (against Direct; against the run's STL-SN), D26 labels, the share of last3 rows
              whose non-seasonal forecast is bit-identical to the run's, and the reading fixed in v2.4

    python -m src.stages.sensitivity2 STAGE --config C --run RUN [--data-dir D] [--group G] [--workers N]

S2 adds no confirmatory hypothesis; the paper reports it as a sensitivity analysis.
"""
from __future__ import annotations

import argparse
import json
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
from src.forecast.registry import config_key
from src.stages import sensitivity as s1
from src.stages.io import StageError, code_commit, load_bundle, sha256_json, utc_now, write_json
from src.stages.tasks import FREQUENCIES, STATISTICAL_CHUNK
from src.validation.gates import load_merged

log = logging.getLogger(__name__)

ROOT = "sensitivity_s2"
GLOBAL_MODELS = ["XGBoost", "NHITS", "PatchTST"]
TUNED_VARIANTS = ["deg0"]                         # a new non-seasonal target: tuned like S1's variants
RUN_CONFIG_VARIANTS = ["last3"]                   # the run's non-seasonal target: the run's configurations
STATISTICAL = [("ETS", "deg0"), ("ETS", "last3"), ("ETS", "stlf"), ("ARIMA", "stlf")]
CONTRASTS = ([(m, v) for m in ["ETS", "XGBoost", "NHITS", "PatchTST"] for v in ["deg0", "last3"]]
             + [("ETS", "stlf"), ("ARIMA", "stlf")])
FROZEN_FORMAT = "rerun-s2-configs/1"


def tune_tasks() -> list[dict]:
    return [{"task_id": f"s2tune|{model}|{frequency}|{variant}", "model": model, "frequency": frequency, "variant": variant,
             "group": s1.group_of(model)}
            for model in GLOBAL_MODELS for frequency in FREQUENCIES for variant in TUNED_VARIANTS]


def eval_tasks(bucket_summary: pd.DataFrame, samples: pd.DataFrame) -> list[dict]:
    features = [f for f in FEATURE_NAMES if f in set(bucket_summary["feature_name"])]
    tasks = []
    for frequency in FREQUENCIES:
        n_chunks = math.ceil(samples.loc[samples["frequency"] == frequency, "unique_id"].nunique() / STATISTICAL_CHUNK)
        for variant in TUNED_VARIANTS + RUN_CONFIG_VARIANTS:
            for feature in features:
                for model in GLOBAL_MODELS:
                    tasks.append({"task_id": f"s2eval|{feature}|{frequency}|{model}|{variant}", "kind": "global",
                                  "group": s1.group_of(model), "frequency": frequency, "feature_name": feature,
                                  "model": model, "variant": variant})
        for model, variant in STATISTICAL:
            for chunk in range(n_chunks):
                tasks.append({"task_id": f"s2stat|{model}|{frequency}|{variant}|chunk{chunk:03d}", "kind": "statistical",
                              "group": "cpu", "frequency": frequency, "chunk": chunk, "model": model, "variant": variant})
    return tasks


def run_entries(run_dir: Path) -> dict:
    """last3 keeps the run's non-seasonal target, so it uses the run's frozen configurations of
    that target (checked against their hash), each marked with the hash it was taken from."""
    record = tuning.load_frozen_configs(Path(run_dir) / "configs_frozen.json")
    out = {}
    for model in GLOBAL_MODELS:
        for frequency in FREQUENCIES:
            entry = dict(record["entries"][config_key(model, frequency, "nonseasonal")])
            for variant in RUN_CONFIG_VARIANTS:
                out[s1.entry_key(model, frequency, variant)] = {**entry, "stl_variant": variant,
                                                                "taken_from_run_configs": record["configs_sha256"]}
    return out


S2 = s1.Study("s2", ROOT, tune_tasks, eval_tasks, FROZEN_FORMAT, run_entries)


# --- analyse ---------------------------------------------------------------------------------------------

def _nonseasonal_hash(components) -> str | None:
    if not isinstance(components, str):
        return None
    parts = [c for c in json.loads(components) if c.get("target") == "nonseasonal"]
    return parts[0]["forecast_hash"] if len(parts) == 1 else None


def identical_share(s2_rows: pd.DataFrame, run_dir: Path, main_seed: int) -> pd.DataFrame:
    """Per model: the share of last3 rows whose non-seasonal forecast is bit-identical to the one in
    the run's STL-SN row of the same model, series, window and feature sample."""
    keys = ["feature_name", "frequency", "model", "unique_id", "window"]
    last3 = s2_rows[(s2_rows["stl_variant"] == "last3") & (s2_rows["status"] == "trained")]
    run = pd.read_parquet(Path(run_dir) / "merged" / "rows.parquet", columns=[*keys, "strategy", "scope", "seed", "components"],
                          filters=[("strategy", "==", "stl_sn"), ("scope", "==", "cohort"),
                                   ("model", "in", sorted(set(last3["model"])))])
    run = run[run["seed"].isna() | (run["seed"] == main_seed)]
    a = last3.assign(h=last3["components"].map(_nonseasonal_hash))[keys + ["h"]]
    b = run.assign(h_run=run["components"].map(_nonseasonal_hash))[keys + ["h_run"]]
    a["feature_name"], b["feature_name"] = a["feature_name"].fillna(""), b["feature_name"].fillna("")
    j = a.merge(b, on=keys, how="left", validate="one_to_one")
    if j["h_run"].isna().any():
        raise StageError(f"{int(j['h_run'].isna().sum())} last3 rows have no STL-SN row in the run")
    j["identical"] = j["h"] == j["h_run"]
    return j.groupby("model", observed=True)["identical"].agg(rows="size", identical_share="mean").reset_index()


def reading(vs_direct: pd.DataFrame, vs_stlsn: pd.DataFrame, default: pd.DataFrame) -> dict:
    """The reading fixed in change log v2.4, from the labels alone."""
    removes = {m: sorted(g.loc[g["label"] == "gain", "variant"].tolist())
               for m, g in vs_stlsn[vs_stlsn["variant"] != "stlf"].groupby("model")}
    d = default.set_index("model")["label"]
    stlf = {m: {"stlf_vs_direct": r["label"], "run_stl_sn_vs_direct": d.get(m)}
            for m, r in vs_direct[vs_direct["variant"] == "stlf"].set_index("model").iterrows()}
    return {"variants_that_remove_part_of_the_harm": removes, "stlf": stlf}


def run_analyse(run_dir: Path, config: dict, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    run_dir = Path(run_dir)
    out = run_dir / ROOT / "analysis"
    if out.exists() and any(out.iterdir()):
        raise StageError(f"{out} is not empty; S2 outputs are never overwritten")
    commit = code_commit()
    bundle = load_bundle(run_dir, config, manifest_path)
    frozen = s1.load_frozen(run_dir, S2)
    prepare = run_dir / "prepare"
    samples, buckets = pd.read_parquet(prepare / "samples.parquet"), pd.read_parquet(prepare / "buckets.parquet")
    tasks = eval_tasks(pd.read_parquet(prepare / "bucket_summary.parquet"), samples)
    s2_rows, facts = s1._s1_rows(run_dir, tasks, ROOT)
    record, rows, failed = load_merged(run_dir, ROW_COLUMNS)          # the run's merged rows, hash-checked
    main_seed = int(config["random_seed"])
    models = sorted({m for m, _ in CONTRASTS})
    keep = rows["model"].isin(models) & (rows["scope"] == "cohort") & rows["strategy"].isin(["direct", "stl_sn"])
    run_rows = rows[keep & (rows["seed"].isna() | (rows["seed"] == main_seed))]

    checks = {"G5_complete": {"expected": len(tasks), "missing": facts["missing"], "changed": facts["changed"]},
              "G4_commit_and_pins": {"commits": facts["commits"], "pinned": facts["pins"], "configs": facts["configs"],
                                     "frozen_configs": frozen["configs_sha256"],
                                     "bundle_matches_run": bundle["bundle_sha256"] == record["provenance"]["bundle_sha256"]}}
    checks["G5_complete"]["passed"] = not facts["missing"] and not facts["changed"]
    checks["G4_commit_and_pins"]["passed"] = (len(facts["commits"]) == 1 and facts["pins"] == [True]
                                              and facts["configs"] == [frozen["configs_sha256"]]
                                              and checks["G4_commit_and_pins"]["bundle_matches_run"])
    trained = s2_rows[s2_rows["status"] == "trained"] if len(s2_rows) else s2_rows
    metrics = ["relnaive_capped", "relnaive", "mase", "smape", "mae", "pocid"]
    nonfinite = int((~np.isfinite(trained[metrics].to_numpy(dtype=float))).any(axis=1).sum()) if len(trained) else 0
    checks["G1_trained_rows"] = {"rows": int(len(s2_rows)), "failed_rows": int(len(s2_rows) - len(trained)),
                                 "listed_task_failures": facts["listed_failures"], "passed": not facts["listed_failures"]}
    checks["G9_finite"] = {"non_finite_rows": nonfinite, "passed": nonfinite == 0}

    s2_inst_rows = s2_rows.assign(strategy="stl_sn_" + s2_rows["stl_variant"].astype(str)) if len(s2_rows) else s2_rows
    both = pd.concat([run_rows, s2_inst_rows[run_rows.columns]], ignore_index=True)
    inst = build_instances(both, failed.iloc[0:0], buckets, main_seed)
    inst = inst[inst["main_seed"]]
    variants = sorted({v for _, v in CONTRASTS})

    def contrast(first: str) -> pd.DataFrame:
        """Instance-level contrasts of the 10 (model, variant) pairs only: a model without a variant
        (ARIMA for deg0, the global models for stlf) would otherwise add Direct rows with no partner."""
        d = pd.concat([strategy_contrast(inst, first, f"stl_sn_{v}").assign(variant=v) for v in variants], ignore_index=True)
        mask = pd.Series(list(zip(d["model"].astype(str), d["variant"]))).isin(set(CONTRASTS)).to_numpy()
        return d[mask].reset_index(drop=True)

    vs_direct, vs_stlsn = contrast("direct"), contrast("stl_sn")
    variant_inst = inst[inst["strategy"].astype(str).str.startswith("stl_sn_")]
    unpaired = int(vs_direct["delta"].isna().sum() - variant_inst["relnaive_capped"].isna().sum())
    unpaired_stlsn = int(vs_stlsn["delta"].isna().sum() - variant_inst["relnaive_capped"].isna().sum())
    checks["G6_pairing"] = {"s2_instances": int(variant_inst["relnaive_capped"].notna().sum()),
                            "unpaired_with_direct": unpaired, "unpaired_with_run_stl_sn": unpaired_stlsn,
                            "passed": unpaired == 0 and unpaired_stlsn == 0}
    passed = all(c["passed"] for c in checks.values())
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "checks.json", {"passed": passed, "checks": checks, "created_at_utc": utc_now()})
    if not passed:
        raise StageError(f"S2 checks failed: {sorted(k for k, v in checks.items() if not v['passed'])}")

    base_seed = int(config["random_seed"])
    wanted = pd.DataFrame(CONTRASTS, columns=["model", "variant"])

    def tests(deltas: pd.DataFrame, family_id: str) -> pd.DataFrame:
        ps = per_series(deltas, ["model", "variant"]).merge(wanted, on=["model", "variant"])
        t = one_sample(ps, ["model", "variant"], base_seed, family_id)
        if len(t) != len(CONTRASTS):
            raise StageError(f"{family_id}: {len(t)} contrasts, expected {len(CONTRASTS)}")
        return t

    t_direct = tests(vs_direct, "S2-vs-direct")          # delta = RN(Direct) - RN(variant): > 0, the variant is better
    t_stlsn = tests(vs_stlsn, "S2-vs-run-stl-sn")        # delta = RN(run's STL-SN) - RN(variant): > 0, the variant is better
    default = one_sample(per_series(strategy_contrast(inst, "direct", "stl_sn"), ["model"]), ["model"], base_seed,
                         "S2-run-stl-sn")
    same = identical_share(s2_rows, run_dir, main_seed)
    read = reading(t_direct, t_stlsn, default)

    tables = {"s2_vs_direct": t_direct, "s2_vs_run_stl_sn": t_stlsn, "s2_run_stl_sn": default, "s2_last3_identical": same}
    (out / "tables").mkdir()
    for name, df in tables.items():
        df.to_csv(out / "tables" / f"{name}.csv", index=False)
    summary = {"status": "sensitivity analysis S2 (protocol change log v2.4); not a confirmatory hypothesis",
               "analysis_code_commit": commit, "created_at_utc": utc_now(), "reading": read,
               "merge_result_sha256": record["result_sha256"], "s2_configs_sha256": frozen["configs_sha256"],
               "tables": {name: sha256_json(df.to_dict(orient="list")) for name, df in tables.items()}}
    write_json(out / "s2.json", summary)
    return {"reading": read, "tests": len(t_direct) + len(t_stlsn)}


def main(argv: list[str] | None = None) -> int:
    from src.utils import read_yaml

    parser = argparse.ArgumentParser(prog="python -m src.stages.sensitivity2", description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["tune", "freeze", "evaluate", "analyse"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--group", choices=["cpu", "gpu"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = read_yaml(args.config)
    if args.stage in ("tune", "evaluate") and (args.group is None or args.data_dir is None):
        parser.error(f"{args.stage} needs --group and --data-dir")
    if args.stage == "tune":
        print(s1.run_tune(args.run, config, args.data_dir, args.group, DEFAULT_MANIFEST, study=S2))
    elif args.stage == "freeze":
        print(s1.run_freeze(args.run, study=S2))
    elif args.stage == "evaluate":
        print(s1.run_evaluate(args.run, config, args.data_dir, args.group, DEFAULT_MANIFEST, workers=args.workers, study=S2))
    else:
        print(run_analyse(args.run, config, DEFAULT_MANIFEST))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
