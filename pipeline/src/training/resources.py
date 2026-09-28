from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CpuQuotaInfo:
    quota_raw: str
    period: int | None
    effective_cores: float


def read_cgroup_cpu_quota(cpu_max_path: str | Path = "/sys/fs/cgroup/cpu.max") -> CpuQuotaInfo:
    path = Path(cpu_max_path)
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception:
        count = float(os.cpu_count() or 1)
        return CpuQuotaInfo(quota_raw="unavailable", period=None, effective_cores=count)

    parts = raw.split()
    if not parts:
        count = float(os.cpu_count() or 1)
        return CpuQuotaInfo(quota_raw="empty", period=None, effective_cores=count)

    if parts[0] == "max":
        count = float(os.cpu_count() or 1)
        period = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        return CpuQuotaInfo(quota_raw=raw, period=period, effective_cores=count)

    try:
        quota = float(parts[0])
        period = int(parts[1]) if len(parts) > 1 else 100000
        effective = quota / max(period, 1)
        return CpuQuotaInfo(quota_raw=raw, period=period, effective_cores=effective)
    except Exception:
        count = float(os.cpu_count() or 1)
        return CpuQuotaInfo(quota_raw=raw, period=None, effective_cores=count)


def detect_effective_cpu_cores() -> float:
    return read_cgroup_cpu_quota().effective_cores


def default_feature_compute_workers() -> int:
    effective = detect_effective_cpu_cores()
    return max(1, min(8, int(round(effective)) - 2))


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def forecast_parallel_config_from_env() -> dict[str, object]:
    effective = detect_effective_cpu_cores()
    defaults = {
        "FORECAST_PARALLEL_ENABLED": False,
        "FORECAST_CPU_TASK_WORKERS": max(1, min(6, int(round(effective)) - 3)),
        "FORECAST_GPU_TASK_WORKERS": 1,
        "FORECAST_STATISTICAL_WORKERS": 4,
        "FORECAST_MLFORECAST_WORKERS": 4,
        "FORECAST_NEURAL_WORKERS": 1,
        "FORECAST_TRANSFORMER_WORKERS": 1,
        "FORECAST_THREADS_PER_CPU_TASK": 1,
        "FORECAST_THREADS_PER_GPU_TASK": 2,
        "FORECAST_CONSERVATIVE_MODE": True,
    }
    out: dict[str, object] = {
        "effective_cpu_cores": effective,
        "cpu_max": read_cgroup_cpu_quota().quota_raw,
        "cpu_period": read_cgroup_cpu_quota().period,
    }
    for key, default in defaults.items():
        raw = os.environ.get(key)
        if isinstance(default, bool):
            out[key.lower()] = _truthy(raw, default)
        else:
            try:
                out[key.lower()] = int(raw) if raw not in {None, ""} else int(default)
            except Exception:
                out[key.lower()] = int(default)
    return out


def safe_task_id(task_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(task_id)).strip("._-")
    return safe or "task"


def task_output_dir(run_root: Path, task_id: str) -> Path:
    return run_root / "task_outputs" / safe_task_id(task_id)


def task_resource_class(task: dict[str, object]) -> str:
    family = str(task.get("model_family", "")).lower()
    if family in {"statistical", "mlforecast"}:
        return "cpu"
    if family in {"neuralforecast", "transformers"}:
        return "gpu"
    return "cpu"


def task_family_bucket(task: dict[str, object]) -> str:
    family = str(task.get("model_family", "")).lower()
    if family == "statistical":
        return "statistical"
    if family == "mlforecast":
        return "mlforecast"
    if family == "neuralforecast":
        return "neural"
    if family == "transformers":
        return "transformer"
    return "cpu"


def classify_tasks(tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    classified = []
    for task in tasks:
        enriched = dict(task)
        enriched["resource_class"] = task_resource_class(enriched)
        enriched["resource_bucket"] = task_family_bucket(enriched)
        enriched["safe_task_id"] = safe_task_id(str(enriched["task_id"]))
        classified.append(enriched)
    return classified


def summarize_task_plan(tasks: list[dict[str, object]], completed_ids: set[str] | None = None) -> dict[str, object]:
    completed_ids = completed_ids or set()
    pending = [task for task in tasks if str(task["task_id"]) not in completed_ids]
    counts: dict[str, int] = {"cpu": 0, "gpu": 0, "statistical": 0, "mlforecast": 0, "neural": 0, "transformer": 0}
    for task in pending:
        counts[task_resource_class(task)] += 1
        counts[task_family_bucket(task)] += 1
    return {
        "total_tasks": len(tasks),
        "completed_tasks": len(completed_ids),
        "queued_tasks": len(pending),
        "resource_class_counts": counts,
    }
