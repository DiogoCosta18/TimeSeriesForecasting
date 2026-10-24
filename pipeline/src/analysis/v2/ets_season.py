"""A3: the seasonality AutoETS chose under Direct, and the statistical family's Delta_STL by that choice.

The run did not store the ETS specification, so AutoETS is fitted again under Direct on every series and window
with the run's arguments (src.forecast.models.statistical_model) and its forecast checked against the stored one
(|difference| <= 1e-6 x mean |y|; more than 0.1% of fits outside stops the step). A series' type is the
seasonality (N, A, M) chosen in at least two of the three windows, "mixed" otherwise.

Outputs in OUT: ``ets_spec.csv`` (series, window, components, check), ``ets_types.csv`` (shares by frequency and
dataset) and Q9: Delta_STL of the statistical family (ARIMA and ETS) in the N, A and M strata, for STL-SN and
STL-AC under the default and periodic settings (k = 12; STL-AC periodic comes from E1), with the mixed stratum
described if it has at least 30 series.

Run:  python -m src.analysis.v2.ets_season --tables TABLES --data-dir DATA --out OUT [--workers N]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.v2 import infer
from src.analysis.v2.a0 import arm, contrast, main_seed
from src.analysis.v2.tables import instances, load, windows

TOLERANCE = 1e-6
MAX_OFF = 0.001
CELLS = [("stl_sn", "default"), ("stl_ac", "default"), ("stl_sn", "periodic")]


def _fit(rows: list[tuple]) -> list[dict]:
    from statsforecast import StatsForecast

    from src.forecast.models import statistical_model

    out = []
    for series_id, window, y, m, h in rows:
        sf = StatsForecast(models=[statistical_model("ETS", m)], freq=1, n_jobs=1)
        sf.fit(df=pd.DataFrame({"unique_id": "s", "ds": range(len(y)), "y": y}))
        yhat = sf.predict(h=h).iloc[:, -1].to_numpy(dtype=float)
        out.append({"series_id": series_id, "window": window, "components": sf.fitted_[0, 0].model_["components"],
                    "yhat": yhat, "scale": float(np.mean(np.abs(y)))})
    return out


def stored_ets(run_dir: Path) -> pd.DataFrame:
    """The run's ETS Direct forecasts (task files)."""
    frames = [pd.read_parquet(p, columns=["unique_id", "window", "yhat"])
              for p in sorted((run_dir / "evaluate").glob("statistical-*/tasks/stat__ETS__*__direct__chunk*.parquet"))]
    return pd.concat(frames, ignore_index=True).rename(columns={"unique_id": "series_id"})


def spec_table(run_dir: Path, data_dir: Path, workers: int) -> pd.DataFrame:
    win = windows(run_dir, data_dir)
    rows = list(zip(win["series_id"], win["window"], win["y_train"], win["m"], win["h"]))
    chunks = [rows[i::max(1, workers)] for i in range(max(1, workers))]
    with ProcessPoolExecutor(max(1, workers)) as pool:
        fits = pd.DataFrame([r for part in pool.map(_fit, chunks) for r in part])
    j = fits.merge(stored_ets(run_dir), on=["series_id", "window"], how="left", suffixes=("", "_stored"), validate="one_to_one")
    if j["yhat_stored"].isna().any():
        raise ValueError("refitted windows without a stored ETS forecast")
    diff = np.array([np.max(np.abs(a - np.asarray(b))) for a, b in zip(j["yhat"], j["yhat_stored"])])
    j["matches"] = diff <= TOLERANCE * j["scale"].to_numpy()
    off = 1 - j["matches"].mean()
    if off > MAX_OFF:
        raise ValueError(f"{off:.2%} of the refitted ETS forecasts differ from the stored ones; check statsforecast")
    j["season"] = j["components"].str[2]
    return j[["series_id", "window", "components", "season", "matches"]].sort_values(["series_id", "window"], ignore_index=True)


def series_type(spec: pd.DataFrame) -> pd.Series:
    def one(s: pd.Series) -> str:
        top = s.value_counts()
        return top.index[0] if top.iloc[0] >= 2 else "mixed"
    return spec.groupby("series_id")["season"].agg(one).rename("ets_type")


def run(tables: Path, data_dir: Path, out: Path, workers: int) -> dict:
    tables, out = Path(tables), Path(out)
    fc, meta, inst = load(tables)
    spec = spec_table(tables.parent, data_dir, workers)
    types = series_type(spec)
    out.mkdir(parents=True, exist_ok=True)
    spec.to_csv(out / "ets_spec.csv", index=False)
    t = meta[["series_id", "freq", "dataset"]].merge(types, left_on="series_id", right_index=True, validate="one_to_one")
    shares = (t.groupby(["freq", "dataset"])["ets_type"].value_counts(normalize=True).rename("share").reset_index()
              .merge(t.groupby(["freq", "dataset"]).size().rename("n").reset_index(), on=["freq", "dataset"]))
    shares.to_csv(out / "ets_types.csv", index=False, float_format="%.10g")

    c = instances(main_seed(fc[(fc["scope"] == "cohort") & (fc["family"] == "statistical") & fc["model"].isin(["ARIMA", "ETS"])
                               & (fc["setting"] != "default_m") & fc["source"].isin(["run", "s3", "s4"])]), inst)
    a = arm(c)
    d = pd.concat([contrast(c, a, "direct/none", f"{s}/{setting}", ["strategy", "setting"]).assign(strategy=s, setting=setting)
                   for s, setting in CELLS], ignore_index=True)
    d["ets_type"] = d["series_id"].astype(str).map(types)
    main = d[d["ets_type"].isin(["N", "A", "M"])]
    X, labels = infer.instance_matrix(main, inst, ["ets_type", "strategy", "setting"])
    q9 = infer.summary(inst, X, labels, 12, "Q9", workers)
    mixed = d[d["ets_type"] == "mixed"]
    if mixed["series_id"].nunique() >= 30:
        s = mixed.groupby(["series_id", "strategy", "setting"], observed=True)["delta"].mean().reset_index()
        q9 = pd.concat([q9, s.groupby(["strategy", "setting"])["delta"].agg(estimate="median", n_series="size").reset_index()
                        .assign(ets_type="mixed", question="Q9 (described only)")], ignore_index=True)
    q9.to_csv(out / "Q9.csv", index=False, float_format="%.10g")
    return {"fits": len(spec), "matching": int(spec["matches"].sum()), "types": types.value_counts().to_dict()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.v2.ets_season", description=__doc__.splitlines()[0])
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    a = parser.parse_args(argv)
    print(run(a.tables, a.data_dir, a.out, a.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
