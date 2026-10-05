"""Small helpers shared by the stages: atomic writes, YAML, the usable CPU count."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


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


def effective_cpu_count(cgroup_root: Path = Path("/sys/fs/cgroup")) -> int:
    """CPUs this process may use: its affinity mask, capped by the container's CPU quota when
    one is set, from cgroup v2 (cpu.max) or v1 (cpu/cpu.cfs_quota_us); inside a container
    os.cpu_count() and the affinity mask report the host's cores."""
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    root = Path(cgroup_root)
    if (root / "cpu.max").exists():
        quota, *period = (root / "cpu.max").read_text(encoding="utf-8").split()
        if quota != "max":
            cores = min(cores, float(quota) / float(period[0] if period else 100000))
    elif (root / "cpu" / "cpu.cfs_quota_us").exists():
        quota = float((root / "cpu" / "cpu.cfs_quota_us").read_text(encoding="utf-8"))
        period = float((root / "cpu" / "cpu.cfs_period_us").read_text(encoding="utf-8"))
        if quota > 0:
            cores = min(cores, quota / period)
    return max(1, int(cores))


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
