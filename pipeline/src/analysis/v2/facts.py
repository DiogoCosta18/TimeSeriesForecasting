"""Single values for the text of the revision, from the canonical tables and the run's records (OUT/facts.csv,
OUT/compute.csv):

- the threshold of 0.01 relative to the mean RelNaive of the best configuration (Table 3) and the median range,
  over the eight cells of Table C3, of the family medians between the three seeds;
- the mean number of feature samples (instances) per series;
- A9: the Ridge penalty chosen in each of the run's studies (median, minimum, maximum);
- A18, A20: hardware and determinism settings of the tasks, and compute hours per study and lane.

Run:  python -m src.analysis.v2.facts --tables TABLES --out OUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.v2.questions import seed_view, stl_contrasts
from src.analysis.v2.a0 import SEEDS
from src.analysis.v2.tables import instances, load
from src.stages.tasks import SEED_CHECK_FEATURE

STUDIES = {"run": "evaluate", "S3": "sensitivity_s3/evaluate", "S4": "sensitivity_s4/evaluate", "S4s": "sensitivity_s4s/evaluate",
           "S5": "sensitivity_s5/evaluate", "S6": "sensitivity_s6/evaluate", "S7": "sensitivity_s7/evaluate",
           "run tuning": "tune", "S4 tuning": "sensitivity_s4/tune", "S7 tuning": "sensitivity_s7/tune"}
GPU_FAMILIES = ("NHITS", "PatchTST", "LSTM", "TFT", "NLinear")


def compute(run_dir: Path) -> pd.DataFrame:
    rows = []
    for study, folder in STUDIES.items():
        for record in sorted((run_dir / folder).glob("**/*.done.json")):
            r = json.loads(record.read_text(encoding="utf-8"))
            lane = "gpu" if any(m in record.name for m in GPU_FAMILIES) else "cpu"
            platform = r.get("platform", {})
            rows.append({"study": study, "lane": lane, "seconds": r.get("compute_seconds", np.nan),
                         "cpu_model": platform.get("cpu_model"), "pinned": platform.get("pinned"),
                         "blas_threads": ",".join(sorted({str(b.get("threads")) for b in platform.get("blas", [])}))})
    return pd.DataFrame(rows)


def run(tables: Path, out: Path) -> pd.DataFrame:
    tables, out = Path(tables), Path(out)
    fc, meta, inst = load(tables)
    run_dir = tables.parent
    f = {}
    cohort = fc[(fc["scope"] == "cohort") & fc["seed"].isin([SEEDS[0], -1]) & (fc["setting"] != "default_m")
                & fc["source"].isin(["run", "s3"])]
    best = cohort.groupby(["model", "strategy"], observed=True)["rn"].mean().min()
    f["v2-thr-relmodel"] = 0.01 / best
    ev = fc[(fc["scope"] == "cohort") & fc["feature_sample"].isin([SEED_CHECK_FEATURE, "none"])
            & (fc["setting"] != "default_m") & fc["source"].isin(["run", "s3"])]
    medians = []
    for sd in SEEDS:
        v = instances(seed_view(ev, sd), inst)
        d = stl_contrasts(v[v["feature_sample"] == SEED_CHECK_FEATURE])
        s = d.groupby(["series_id", "family", "strategy"], observed=True)["delta"].mean()
        medians.append(s.groupby(level=["family", "strategy"], observed=True).median().rename(sd))
    m = pd.concat(medians, axis=1)
    f["v2-thr-seed-spread"] = float((m.max(axis=1) - m.min(axis=1)).median())
    f["v2-mean-instances-per-series"] = len(inst) / inst["series_id"].nunique()
    configs = json.loads((run_dir / "configs_frozen.json").read_text(encoding="utf-8"))["entries"]
    alphas = [e["params"]["alpha"] for k, e in configs.items() if k.startswith("Ridge|")]
    f["v2-a9-ridge-alpha-median"], f["v2-a9-ridge-alpha-min"], f["v2-a9-ridge-alpha-max"] = (
        float(np.median(alphas)), float(min(alphas)), float(max(alphas)))
    f["v2-a9-ridge-studies"] = len(alphas)
    c = compute(run_dir)
    hours = c.groupby(["study", "lane"])["seconds"].sum().div(3600).rename("hours").reset_index()
    out.mkdir(parents=True, exist_ok=True)
    hours.to_csv(out / "compute.csv", index=False, float_format="%.4g")
    f["v2-a18-cpu-models"] = ";".join(sorted({x for x in c["cpu_model"] if x}))
    f["v2-a18-all-pinned"] = bool(c["pinned"].dropna().all())
    f["v2-a18-blas-threads"] = ";".join(sorted({x for x in c["blas_threads"] if x}))
    facts = pd.DataFrame({"key": list(f), "value": list(f.values())})
    facts.to_csv(out / "facts.csv", index=False)
    return facts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.v2.facts", description=__doc__.splitlines()[0])
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    a = parser.parse_args(argv)
    print(run(a.tables, a.out).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
