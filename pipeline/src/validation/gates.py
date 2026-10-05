"""Output gates G1-G13 (protocol Section 7.4).

``run_gates`` checks the merged result bundle together with the prepare bundle and the
frozen configurations, and writes ``merged/gates.json``: for every gate whether it
passed and the numbers behind it, tied to the merge's result hash. The analysis runs
only on a bundle whose report passed and still matches it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data.eligibility import eligibility_length
from src.data.frozen import DEFAULT_MANIFEST, sha256_file
from src.data.load_m_datasets import frozen_data_provenance
from src.forecast.engine import METRICS
from src.forecast.metrics import RELNAIVE_CAP
from src.forecast.registry import STRATEGY_TARGETS
from src.stages.io import StageError, code_commit, load_bundle, read_json, sha256_json, utc_now, write_json
from src.stages.merge import check_twins, run_provenance

GATES_FORMAT = "rerun-v2-gates-1"
STATISTICAL_FAILURE_MAX = 0.005   # G5
CAPPED_SHARE_MAX = 0.02           # G9
TIME_RATIO_RANGE = (2.0, 4.0)     # G10
KS_MIN_P = 0.01                   # G11
STL_TWIN_KEYS = ["feature_name", "frequency", "family", "model", "scope", "seed", "unique_id", "window"]
TASK_WINDOW_KEYS = ["feature_name", "frequency", "family", "model", "scope", "seed", "window"]


def _py(value):
    """JSON-safe Python scalars and containers."""
    if isinstance(value, dict):
        return {str(k): _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_py(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    return value


def load_merged(run_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """The merge record and its two tables, after checking they are unchanged."""
    merged = Path(run_dir) / "merged"
    record = read_json(merged / "merge.json")
    body = {k: v for k, v in record.items() if k not in ("result_sha256", "created_at_utc", "merge_code_commit")}
    if sha256_json(body) != record["result_sha256"]:
        raise StageError("merge.json does not match its result hash")
    for entry in record["files"].values():
        if sha256_file(merged / entry["file"]) != entry["sha256"]:
            raise StageError(f"merged/{entry['file']} changed since the merge")
    return record, pd.read_parquet(merged / "rows.parquet"), pd.read_parquet(merged / "failed_rows.parquet")


def _key_frame(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Keys with missing values made explicit, so statistical rows (no feature, seed) pair."""
    out = df[keys].copy()
    for column in ("feature_name", "scope"):
        if column in out:
            out[column] = out[column].astype(object).where(out[column].notna(), "-")
    if "seed" in out:
        out["seed"] = out["seed"].astype("Int64").fillna(-1).astype("int64")
    return out


def g1_trained_rows_only(rows, failed, record) -> dict:
    expected = rows["strategy"].map(lambda s: len(STRATEGY_TARGETS[s]))
    components = rows["components"].str.count('"target":')
    return {
        "passed": bool((rows["status"] == "trained").all() and (components == expected).all()
                       and (failed["status"] == "failed").all() and failed["failure"].notna().all()
                       and (failed["family"] == "statistical").all()),
        "rows": len(rows), "failed_rows": len(failed),
        "rows_with_wrong_component_count": int((components != expected).sum()),
        "listed_task_failures": len(record["failures"]["listed"]),
        "forecast_hashes_verified_by_merge": record["rows"]["trained"],
    }


def g2_g3_g13_hashes(rows, failed, main, bundle, manifest_path) -> dict:
    both = pd.concat([rows, failed], ignore_index=True)
    manifests = sorted(both["data_manifest_sha256"].unique())
    configs = sorted(both["configs_sha256"].unique())
    data_block = frozen_data_provenance(manifest_path)
    return {
        "G2": {"passed": manifests == [main["data_manifest_sha256"]], "manifest_hashes": manifests},
        "G3": {"passed": configs == [main["configs_sha256"]], "configuration_hashes": configs},
        "G13": {"passed": bundle["data"] == data_block and manifests == [data_block["manifest_sha256"]],
                "bundle_data_matches_manifest": bundle["data"] == data_block,
                "frozen_files": {k: v["sha256"] for k, v in data_block["frozen_files"].items()}},
    }


def g4_commit_and_environment(rows, failed, record, main) -> dict:
    both = pd.concat([rows, failed], ignore_index=True)
    envs = sorted(both["environment_lock_sha256"].unique())
    by_task = both.groupby("task_id")["code_commit"].unique()
    off = {tid: list(c) for tid, c in by_task.items() if list(c) != [main["code_commit"]]}
    d16 = record["d16_reruns"]
    unexplained = {tid: c for tid, c in off.items() if tid not in d16 or c != [d16[tid]["fix_commit"]]}
    platforms = {**{tid: e["platform"] for tid, e in record["tasks"].items()}, **record["study_platforms"]}
    not_pinned = sorted(tid for tid, p in platforms.items() if not p["pinned"])
    cpu_by_shard = {}
    for e in record["tasks"].values():
        cpu_by_shard.setdefault(e["shard"], set()).add(e["platform"]["cpu_model"])
    mixed = {s: sorted(c) for s, c in cpu_by_shard.items() if s.startswith("statistical") and len(c) > 1}
    return {"passed": envs == [main["environment_lock_sha256"]] and not unexplained and not not_pinned and not mixed,
            "code_commit": main["code_commit"], "environment_lock_hashes": envs,
            "d16_reruns": {tid: d16[tid]["fix_commit"] for tid in sorted(off) if tid in d16},
            "unexplained_commits": unexplained, "tasks_without_pinned_platform": not_pinned,
            "statistical_shards_on_several_cpu_models": mixed,
            "cpu_models_by_shard": {s: sorted(c) for s, c in sorted(cpu_by_shard.items())}}


def g5_grid_complete(rows, failed, record) -> dict:
    grid = record["grid"]
    statistical = (rows["family"] == "statistical").sum() + len(failed)
    share = len(failed) / statistical if statistical else 0.0
    return {"passed": not grid["missing"] and not grid["listed_failures"] and share <= STATISTICAL_FAILURE_MAX,
            "expected": grid["expected"], "completed": grid["completed"], "missing": grid["missing"],
            "listed_failures": grid["listed_failures"], "statistical_failed_share": share,
            "statistical_failed_by_model": failed.groupby(["model", "strategy"]).size().to_dict() if len(failed) else {}}


def g6_pairing(rows, failed) -> dict:
    both = pd.concat([rows, failed], ignore_index=True)
    keys = _key_frame(both, STL_TWIN_KEYS + ["strategy"])
    direct = keys[keys["strategy"] == "direct"].drop(columns="strategy")
    stl = keys[keys["strategy"] != "direct"].drop(columns="strategy")
    duplicated_direct = int(direct.duplicated().sum())
    paired = stl.merge(direct.drop_duplicates().assign(_twin=True), on=STL_TWIN_KEYS, how="left")
    missing = int(paired["_twin"].isna().sum())
    twins = check_twins(rows)
    return {"passed": missing == 0 and duplicated_direct == 0 and twins["passed"],
            "stl_rows": int(len(stl)), "stl_without_direct_twin": missing, "duplicated_direct_rows": duplicated_direct,
            "tercile": twins}


def g7_no_leakage(tables, frozen) -> dict:
    first = tables["cutoffs"].query("window == 0").set_index("unique_id")["train_end_idx"]
    features = tables["features"][tables["features"]["eligible"]].set_index("unique_id")["history_end_t"]
    tuning = tables["tuning_set"].set_index("unique_id")["tuning_validation_end_t"]
    late_features = int((features > first.reindex(features.index)).sum() + first.reindex(features.index).isna().sum())
    late_tuning = int((tuning > first.reindex(tuning.index)).sum() + first.reindex(tuning.index).isna().sum())
    unchecked = sorted(k for k, e in frozen["entries"].items() if e.get("g7_tuning_end_equals_first_cutoff") is not True)
    return {"passed": late_features == 0 and late_tuning == 0 and not unchecked,
            "features_after_first_cutoff": late_features, "tuning_after_first_cutoff": late_tuning,
            "studies_without_g7_check": unchecked}


def g8_minimum_history(tables, config) -> dict:
    elig = tables["eligibility"]
    eligible = set(elig.loc[elig["eligible"], "unique_id"])
    short, too_short_cutoffs = 0, 0
    for frequency, fcfg in config["frequencies"].items():
        L = eligibility_length(int(fcfg["season_length"]), int(fcfg["horizon"]))
        e = elig[(elig["frequency"] == frequency) & elig["eligible"]]
        short += int((e["history_length"] < L).sum())
        c = tables["cutoffs"][(tables["cutoffs"]["frequency"] == frequency) & (tables["cutoffs"]["window"] == 0)]
        too_short_cutoffs += int((c["train_end_idx"] < L).sum())
    outside = {name: int((~tables[name]["unique_id"].isin(eligible)).sum()) for name in ("samples", "tuning_set", "cutoffs")}
    return {"passed": short == 0 and too_short_cutoffs == 0 and not any(outside.values()),
            "eligible_below_L": short, "window0_below_L": too_short_cutoffs, "ineligible_downstream": outside}


def g9_finite_metrics(rows) -> dict:
    finite = np.isfinite(rows[METRICS].to_numpy(dtype=float))
    capped = rows["relnaive"] > RELNAIVE_CAP
    share = float(capped.mean()) if len(rows) else 0.0
    return {"passed": bool(finite.all()) and share <= CAPPED_SHARE_MAX,
            "non_finite_values": int((~finite).sum()), "capped_share": share,
            "capped_share_by_family": capped.groupby(rows["family"]).mean().to_dict()}


def g10_time_ratio(rows) -> dict:
    glob = rows[rows["family"] != "statistical"]
    per_window = _key_frame(glob, TASK_WINDOW_KEYS + ["strategy"]).assign(fit_seconds=glob["fit_seconds"].to_numpy())
    per_window = per_window.groupby(TASK_WINDOW_KEYS + ["strategy"], sort=True)["fit_seconds"].first().unstack("strategy")
    medians = {}
    for fam, g in per_window.groupby(level="family"):
        if {"stl_ac", "stl_sn"} <= set(g.columns):
            ratio = (g["stl_ac"] / g["stl_sn"]).dropna()
            medians[fam] = float(ratio.median()) if len(ratio) else None
        else:
            medians[fam] = None
    lo, hi = TIME_RATIO_RANGE
    return {"passed": bool(medians) and all(v is not None and lo <= v <= hi for v in medians.values()),
            "median_stl_ac_over_stl_sn": medians, "range": [lo, hi]}


def g11_sampling(tables) -> dict:
    features = tables["features"][tables["features"]["eligible"]]
    samples = tables["samples"]
    tests, failing = [], []
    for (frequency, feature, source), s in samples.groupby(["frequency", "feature_name", "source_dataset"], sort=True):
        pool = features[(features["frequency"] == frequency) & (features["source_dataset"] == source)][feature]
        p = float(ks_2samp(s["feature_value"], pool).pvalue)
        tests.append({"frequency": frequency, "feature": feature, "source": source, "n_sample": len(s), "n_pool": len(pool), "p": p})
        if source.startswith("M4") and p < KS_MIN_P:
            failing.append(f"{feature}|{frequency}|{source}")
    return {"passed": not failing, "min_p": KS_MIN_P, "failing": failing, "tests": tests}


def g12_buckets(tables, config) -> dict:
    min_share = float(config["buckets"]["min_share"])
    summary = tables["bucket_summary"].set_index(["frequency", "feature_name"])
    inconsistent, degenerate = [], []
    for (frequency, feature), g in tables["buckets"].groupby(["frequency", "feature_name"], sort=True):
        sizes = g["feature_tercile_bucket"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
        shares = sizes / sizes.sum()
        flag = bool((shares < min_share).any())
        populated = ",".join(b for b in ["Low", "Medium", "High"] if sizes[b] > 0)
        row = summary.loc[(frequency, feature)]
        if bool(row["degenerate"]) != flag or row["populated_buckets"] != populated:
            inconsistent.append(f"{feature}|{frequency}")
        if flag:
            degenerate.append({"feature": feature, "frequency": frequency, "sizes": sizes.to_dict()})
    return {"passed": not inconsistent and len(summary) == tables["buckets"].groupby(["frequency", "feature_name"]).ngroups,
            "inconsistent_summary": inconsistent, "degenerate": degenerate}


def run_gates(run_dir: Path, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    run_dir = Path(run_dir)
    record, rows, failed = load_merged(run_dir)
    config = read_json(run_dir / "prepare" / "bundle.json")["config"]
    bundle = load_bundle(run_dir, config, manifest_path)
    main, frozen = run_provenance(run_dir, bundle, manifest_path)
    if main != record["provenance"]:
        raise StageError("the merged bundle was built with other provenance than the run's")
    tables = {name: pd.read_parquet(run_dir / "prepare" / entry["file"]) for name, entry in bundle["files"].items()}
    hashes = g2_g3_g13_hashes(rows, failed, main, bundle, manifest_path)
    gates = {
        "G1": g1_trained_rows_only(rows, failed, record),
        "G2": hashes["G2"],
        "G3": hashes["G3"],
        "G4": g4_commit_and_environment(rows, failed, record, main),
        "G5": g5_grid_complete(rows, failed, record),
        "G6": g6_pairing(rows, failed),
        "G7": g7_no_leakage(tables, frozen),
        "G8": g8_minimum_history(tables, config),
        "G9": g9_finite_metrics(rows),
        "G10": g10_time_ratio(rows),
        "G11": g11_sampling(tables),
        "G12": g12_buckets(tables, config),
        "G13": hashes["G13"],
    }
    gates = _py(gates)
    report = {"format": GATES_FORMAT, "merge_result_sha256": record["result_sha256"], "gates": gates,
              "passed": all(g["passed"] for g in gates.values()), "created_at_utc": utc_now(),
              "gates_code_commit": code_commit()}
    write_json(run_dir / "merged" / "gates.json", report)
    return report
