"""Pilot measurement (protocol Section 8.2, item 4): peak GPU memory of every neural and
transformer model over its whole discrete search grid, at the largest input and batch sizes.

    python scripts/gpu_memory_probe.py --config CONFIG --run RUN --data-dir DATA [--steps 20]

Each model is fitted for a few steps on the cohort pool of the run's first feature sample,
once per frequency and combination of its discrete parameters; the learning rate and the
scaler do not change the memory needed, and dropout is fixed at 0. An out-of-memory error
is recorded as a result: narrowing a search space for memory is a permitted pilot
adjustment. Writes RUN/pilot/gpu_memory.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from neuralforecast import NeuralForecast  # noqa: E402

from src.data.frozen import DEFAULT_MANIFEST, load_manifest  # noqa: E402
from src.data.load_m_datasets import load_dataset_pair  # noqa: E402
from src.forecast import spaces  # noqa: E402
from src.forecast.registry import FAMILY, GLOBAL_MODELS  # noqa: E402
from src.utils import read_yaml  # noqa: E402

FIXED = {"scaler_type": "identity", "learning_rate": 1e-3, "dropout": 0.0}
NOT_ENUMERATED = {"input_size", "batch_size", "scaler_type", "max_steps", "dropout"}


class _Grid:
    """Optuna-like trial: records the discrete choices and answers from ``values``."""

    def __init__(self, values=None):
        self.values, self.choices = values or {}, {}

    def suggest_categorical(self, name, choices):
        self.choices[name] = list(choices)
        return self.values.get(name, choices[0])

    def suggest_float(self, name, low, high, log=False):
        return FIXED[name]


def grid(model: str, m: int):
    """Every combination of the model's discrete parameters, at the largest input and batch size."""
    probe = _Grid()
    spaces.sample_neural(model, probe, m)
    largest = {"input_size": max(probe.choices["input_size"]), "batch_size": max(probe.choices["batch_size"])}
    free = {k: v for k, v in probe.choices.items() if k not in NOT_ENUMERATED}
    for combo in itertools.product(*free.values()):
        yield spaces.sample_neural(model, _Grid({**FIXED, **largest, **dict(zip(free, combo))}), m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible")
    config = read_yaml(args.config)
    samples = pd.read_parquet(args.run / "prepare" / "samples.parquet")
    cutoffs = pd.read_parquet(args.run / "prepare" / "cutoffs.parquet")
    results = []
    for frequency, fcfg in config["frequencies"].items():
        m, h = int(fcfg["season_length"]), int(fcfg["horizon"])
        sub = samples[samples["frequency"] == frequency]
        pool = sorted(sub.loc[sub["feature_name"] == sub["feature_name"].iloc[0], "unique_id"])
        data = load_dataset_pair({"frequency": frequency, **fcfg}, args.data_dir, load_manifest(DEFAULT_MANIFEST))
        ends = cutoffs[(cutoffs["frequency"] == frequency) & (cutoffs["window"] == 0)].set_index("unique_id")["train_end_idx"]
        data = data[data["unique_id"].isin(pool)]
        train = data[data["t"] <= data["unique_id"].map(ends)][["unique_id", "t", "y"]].rename(columns={"t": "ds"})
        for model in [g for g in GLOBAL_MODELS if FAMILY[g] in ("neural", "transformer")]:
            for params in grid(model, m):
                kwargs = spaces.neural_kwargs(model, params, h, int(config["random_seed"]), max_steps=args.steps,
                                              accelerator="gpu")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                row = {"frequency": frequency, "model": model, "params": params, "pool_series": len(pool)}
                try:
                    NeuralForecast(models=[spaces.neural_class(model)(**kwargs)], freq=1).fit(train, val_size=0)
                    row.update(status="ok", peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                               peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20)
                except torch.cuda.OutOfMemoryError as exc:  # a result of the measurement, not a substitute
                    row.update(status="out_of_memory", error=str(exc)[:300])
                row["seconds"] = time.perf_counter() - start
                results.append(row)
                print(json.dumps({"frequency": frequency, "model": model, "status": row["status"],
                                  "peak_reserved_mb": round(row.get("peak_reserved_mb", -1))}), flush=True)
    props = torch.cuda.get_device_properties(0)
    summary = {}
    for (frequency, model), g in pd.DataFrame(results).groupby(["frequency", "model"]):
        ok = g[g["status"] == "ok"]
        top = ok.loc[ok["peak_reserved_mb"].idxmax()] if len(ok) else None
        summary[f"{model}|{frequency}"] = {"combinations": len(g), "out_of_memory": int((g["status"] != "ok").sum()),
                                           "max_peak_reserved_mb": None if top is None else float(top["peak_reserved_mb"]),
                                           "at": None if top is None else top["params"]}
    out = args.run / "pilot" / "gpu_memory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gpu": props.name, "total_mb": props.total_memory / 2**20, "steps": args.steps,
                               "summary": summary, "fits": results}, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
