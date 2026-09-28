from __future__ import annotations

import subprocess


def gpu_memory() -> dict[str, int | None]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        used, total = [int(x.strip()) for x in out.split(",")[:2]]
        return {"gpu_memory_used_mb": used, "gpu_memory_total_mb": total}
    except Exception:
        return {"gpu_memory_used_mb": None, "gpu_memory_total_mb": None}

