"""The numeric platform a process runs on, and whether the pins of src/__init__.py took effect.

Recorded in every completion record (stages) and checked by gate G4: the pins only work if
they are set before numpy loads OpenBLAS, so the record reads what the loaded libraries
report rather than what the environment says.
"""
from __future__ import annotations

import functools
import platform as _platform
from pathlib import Path

from src import PINNED_ENV


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return _platform.processor() or "unknown"


@functools.lru_cache(maxsize=1)
def platform_info() -> dict:
    import numba
    import numpy  # noqa: F401  (loads OpenBLAS, so that it is reported)
    import scipy.linalg  # noqa: F401
    from threadpoolctl import threadpool_info

    blas = [{"library": d.get("prefix"), "architecture": d.get("architecture"), "threads": d.get("num_threads")}
            for d in threadpool_info() if d.get("internal_api") == "openblas"]
    openmp = [d.get("num_threads") for d in threadpool_info() if d.get("internal_api") == "openmp"]
    pinned = (bool(blas) and all(b["architecture"] == PINNED_ENV["OPENBLAS_CORETYPE"] and b["threads"] == 1 for b in blas)
              and all(t == 1 for t in openmp) and numba.config.CPU_NAME == PINNED_ENV["NUMBA_CPU_NAME"])
    return {"cpu_model": _cpu_model(), "machine": _platform.machine(), "libc": "-".join(_platform.libc_ver()),
            "blas": blas, "openmp_threads": openmp, "numba_cpu_name": numba.config.CPU_NAME, "pinned": pinned}
