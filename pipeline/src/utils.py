from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dirs(root: Path) -> None:
    for rel in [
        "sampling",
        "features",
        "cv",
        "final/model_artifacts",
        "optuna",
        "finetuning",
        "checkpoints",
        "logs",
        "reports",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    tmp = path.with_name(f".{path.name}.tmp")
    if suffix == ".parquet":
        df.to_parquet(tmp, index=False)  # a failure is an error, never a CSV under a .parquet name
    elif suffix == ".csv":
        df.to_csv(tmp, index=False)
    else:
        df.to_json(tmp, orient="records", lines=True)
    os.replace(tmp, path)


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "statsmodels",
        "ruptures",
        "datasetsforecast",
        "statsforecast",
        "mlforecast",
        "neuralforecast",
        "torch",
        "optuna",
    ]:
        try:
            mod = __import__(name)
            packages[name] = getattr(mod, "__version__", "installed")
        except Exception as exc:
            packages[name] = f"unavailable: {exc.__class__.__name__}"
    return {
        "created_at_utc": utc_now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
    }


def git_snapshot(cwd: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=str(cwd), text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": run(["git", "status", "--porcelain"]) not in {None, ""},
    }


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return default
