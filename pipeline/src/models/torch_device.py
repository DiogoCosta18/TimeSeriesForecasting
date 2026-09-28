from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache


@lru_cache(maxsize=1)
def _cuda_is_usable() -> bool:
    code = (
        "import torch; "
        "ok=torch.cuda.is_available() and torch.cuda.device_count()>0; "
        "torch.cuda.get_device_capability(0) if ok else None; "
        "raise SystemExit(0 if ok else 1)"
    )
    try:
        return subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=20,
        ).returncode == 0
    except Exception:
        return False


def lightning_trainer_device_kwargs() -> dict[str, object]:
    if _cuda_is_usable():
        return {"accelerator": "gpu", "devices": 1}
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return {"accelerator": "cpu", "devices": 1}
