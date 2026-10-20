"""Exploratory analysis E10, evolving seasonality with periodic STL (protocol change log v2.12 (g)): NOT pre-registered.

phi4' is phi4 (D4) with every rolling window decomposed by periodic STL (as in S1) instead of the STL
of D9, on the history before the first test window of every sampled series; its terciles are cut per
frequency over the sampled series.

- Control: phi4 recomputed by the same code with the STL of D9 equals the stored phi4 of every
  sampled series exactly (NaN where the stored value is undefined). If it does not, E10 stops after
  writing the control.
- Analysis: R3 (src.analysis.review) with phi4' beside phi4 and phi7: Delta_STL by tercile with the
  Jonckheere-Terpstra test, Spearman and partial Spearman correlations (phi4' given phi7).

    python -m src.analysis.exploratory3 --config C --run RUN --data-dir D [--strength E7_TABLE] [--statistical-stl p1]
                                        [--jobs N] [--out OUT]

OUT defaults to RUN/exploratory3_<commit>. Without E7's table, the strength is computed as E7 computes it.
"""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import review
from src.analysis.data import AnalysisError, load_gated
from src.data.frozen import DEFAULT_MANIFEST, load_manifest
from src.data.load_m_datasets import load_dataset_pair
from src.features.errors import FeatureUndefined
from src.features.mstl_features import evolving_seasonality_feature
from src.stages.io import code_commit, sha256_json, utc_now, write_json

STATUS = "exploratory: not pre-registered (protocol change log v2.12 (g); Section 5 is the confirmatory plan)"
EV = review.EV
EVP = review.EVP


def _phi4(y: np.ndarray, m: int, periodic: bool) -> tuple[float, str]:
    try:
        value, _ = evolving_seasonality_feature(y, m, periodic=periodic)
    except FeatureUndefined as exc:
        return math.nan, f"undefined:{exc.reason}"
    except Exception as exc:  # recorded, never replaced by a value (as the run's features)
        return math.nan, f"error:{type(exc).__name__}: {exc}"[:300]
    return (float(value), "ok") if math.isfinite(value) else (math.nan, "error:non_finite_value")


def _one(payload: tuple) -> dict:
    uid, frequency, y, m = payload
    run, run_flag = _phi4(y, m, periodic=False)
    periodic, flag = _phi4(y, m, periodic=True)
    return {"unique_id": uid, "frequency": frequency, "phi4_recomputed": run, "phi4_recomputed_flag": run_flag,
            EVP: periodic, "periodic_flag": flag, "history_length": len(y)}


def phi4_table(config: dict, data_dir: Path, features: pd.DataFrame, ids: pd.DataFrame, jobs: int = 1,
               manifest_path: Path = DEFAULT_MANIFEST) -> pd.DataFrame:
    """phi4 recomputed and phi4' of every sampled series, on its history before the first test window."""
    ends = features.set_index(["unique_id", "frequency"])["history_end_t"]
    payloads = []
    for frequency, want in ids.groupby("frequency", sort=True):
        fcfg = config["frequencies"][frequency]
        m = int(fcfg["season_length"])
        data = load_dataset_pair({"frequency": frequency, **fcfg}, data_dir, load_manifest(manifest_path))
        data = data[data["unique_id"].isin(set(want["unique_id"]))]
        for uid, g in data.groupby("unique_id", sort=True):
            y = g.sort_values("t")["y"].to_numpy(dtype=float)[: int(ends.loc[(uid, frequency)])]
            payloads.append((uid, frequency, y, m))
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as pool:
            rows = list(pool.map(_one, payloads, chunksize=64))
    else:
        rows = [_one(p) for p in payloads]
    table = pd.DataFrame(rows)
    stored = features[["unique_id", "frequency", EV]].rename(columns={EV: "phi4_stored"})
    return table.merge(stored, on=["unique_id", "frequency"], how="left", validate="one_to_one")


def control(table: pd.DataFrame, n_sampled: int) -> dict:
    a, b = table["phi4_recomputed"].to_numpy(), table["phi4_stored"].to_numpy()
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    return {"series": int(len(table)), "sampled_series": int(n_sampled), "identical": int(same.sum()),
            "max_abs_difference": float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else 0.0,
            "periodic_undefined": int(table[EVP].isna().sum()),
            "passed": len(table) == n_sampled and bool(same.all())}


def run_exploratory3(run_dir: Path, config: dict, data_dir: Path, strength: Path | None, out: Path | None = None,
                     statistical_stl: str = "run", jobs: int = 1, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    commit = code_commit()
    out = Path(out) if out is not None else Path(run_dir) / f"exploratory3_{commit[:7]}"
    if out.exists() and any(out.iterdir()):
        raise AnalysisError(f"{out} is not empty; exploratory outputs are never overwritten")
    d = load_gated(run_dir, manifest_path)
    if statistical_stl == "p1":
        from src.stages.sensitivity3 import main_specification
        d = main_specification(d)
    features = d.tables["features"]
    ids = d.tables["samples"][["unique_id", "frequency"]].drop_duplicates()
    table = phi4_table(config, data_dir, features, ids, jobs, manifest_path)
    check = control(table, len(ids))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "e10_phi4.csv", index=False)
    write_json(out / "control.json", {**check, "created_at_utc": utc_now(), "code_commit": commit})
    if not check["passed"]:
        raise AnalysisError(f"E10 control failed: phi4 is not reproduced ({check}); not analysed")

    if strength is not None:
        strength_table = pd.read_csv(strength)
    else:
        from src.analysis.exploratory2 import strength_table as e7_strength
        strength_table = e7_strength(config, data_dir, features, ids, manifest_path)
    x, cuts = review.cells(features, d.tables["bucket_summary"], strength_table, table)
    terc, corr, fam_terc = review.r3(d.main, x, d.main_seed)
    outputs = {"e10_terciles": terc, "e10_correlations": corr, "e10_family_terciles": fam_terc, "e10_cuts": cuts}
    for name, df in outputs.items():
        df.to_csv(out / f"{name}.csv", index=False)
    record = {"status": STATUS, "code_commit": commit, "created_at_utc": utc_now(), "statistical_stl": statistical_stl,
              "strength_table": str(strength) if strength is not None else "computed as E7 (no table given)",
              "merge_result_sha256": d.record["result_sha256"],
              "control": check, "tables": {name: sha256_json(df.to_dict(orient="list"))
                                           for name, df in {"e10_phi4": table, **outputs}.items()}}
    write_json(out / "exploratory3.json", record)
    return {"out": str(out), "control_passed": True, "tables": len(outputs) + 1}


def main(argv: list[str] | None = None) -> int:
    from src.utils import read_yaml

    parser = argparse.ArgumentParser(prog="python -m src.analysis.exploratory3", description=STATUS)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--strength", type=Path, help="E7's e7_strength.csv (default: computed as E7 does)")
    parser.add_argument("--statistical-stl", choices=["run", "p1"], default="run")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    print(run_exploratory3(args.run, read_yaml(args.config), args.data_dir, args.strength, args.out,
                           args.statistical_stl, args.jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
