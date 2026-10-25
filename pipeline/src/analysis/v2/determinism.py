"""A18: are the neural fits deterministic given the seed on one machine? NHITS, PatchTST and LSTM under Direct on the
monthly evolving-seasonality pool, window 0, the main seed and the frozen configurations, each fitted twice; the
maximum absolute and relative difference of the forecasts, with the machine and the library versions.

Run:  python -m src.analysis.v2.determinism --run RUN --data-dir DATA --out OUT
"""
from __future__ import annotations

import argparse
import json
import platform
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.frozen import DEFAULT_MANIFEST, load_manifest
from src.data.load_m_datasets import load_dataset_pair
from src.forecast import models
from src.forecast.registry import config_key
from src.stages.io import read_json
from src.stages.tasks import SEED_CHECK_FEATURE

MODELS = ["NHITS", "PatchTST", "LSTM"]


def run(run_dir: Path, data_dir: Path, out: Path) -> dict:
    run_dir = Path(run_dir)
    config = read_json(run_dir / "prepare" / "bundle.json")["config"]
    fcfg = config["frequencies"]["monthly"]
    m, h, seed = int(fcfg["season_length"]), int(fcfg["horizon"]), int(config["random_seed"])
    samples = pd.read_parquet(run_dir / "prepare" / "samples.parquet")
    ids = set(samples[(samples["frequency"] == "monthly") & (samples["feature_name"] == SEED_CHECK_FEATURE)]["unique_id"])
    cut = pd.read_parquet(run_dir / "prepare" / "cutoffs.parquet")
    cut = cut[(cut["frequency"] == "monthly") & (cut["window"] == 0) & cut["unique_id"].isin(ids)].set_index("unique_id")["train_end_idx"]
    data = load_dataset_pair({"frequency": "monthly", **fcfg}, data_dir, load_manifest(DEFAULT_MANIFEST))
    data = data[data["unique_id"].isin(ids)]
    train = pd.concat([g.sort_values("t").iloc[: int(cut[u])][["unique_id", "t", "y"]].rename(columns={"t": "ds"})
                       for u, g in data.groupby("unique_id")], ignore_index=True)
    entries = json.loads((run_dir / "configs_frozen.json").read_text(encoding="utf-8"))["entries"]
    rows = []
    for model in MODELS:
        e = entries[config_key(model, "monthly", "raw")]
        fits = [models.fit_predict_global(model, e["params"], train, h, m, seed, trained_steps=e.get("trained_steps"),
                                          config_key=config_key(model, "monthly", "raw")) for _ in range(2)]
        a = np.stack([fits[0][u].yhat for u in sorted(ids)])
        b = np.stack([fits[1][u].yhat for u in sorted(ids)])
        diff = np.abs(a - b)
        rows.append({"model": model, "series": len(ids), "max_abs_diff": float(diff.max()),
                     "max_rel_diff": float((diff / np.maximum(np.abs(a), 1e-12)).max()), "identical": bool((diff == 0).all())})
    import torch
    record = {"models": rows, "machine": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
              "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "libraries": {p: metadata.version(p) for p in ("neuralforecast", "pytorch_lightning", "statsforecast", "mlforecast")}}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "determinism.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.analysis.v2.determinism", description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    a = parser.parse_args(argv)
    print(json.dumps(run(a.run, a.data_dir, a.out)["models"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
