"""Numeric pins (src/__init__.py) and the platform record (protocol change log v1.9; gate G4)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from src import PINNED_ENV

ROOT = Path(__file__).resolve().parents[1]
REPORT = "import json; from src.numeric_platform import platform_info; print(json.dumps(platform_info()))"


def _platform(code: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in PINNED_ENV}  # nothing inherited from this process
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_pins_take_effect_when_src_is_imported_first():
    info = _platform("import src; " + REPORT)
    assert info["pinned"] and info["numba_cpu_name"] == PINNED_ENV["NUMBA_CPU_NAME"]
    assert info["blas"] and all(b["architecture"] == "Haswell" and b["threads"] == 1 for b in info["blas"])
    assert all(t == 1 for t in info["openmp_threads"]) and info["cpu_model"] and info["libc"].startswith("glibc")


def test_pins_are_reported_as_not_in_effect_when_numpy_loads_first():
    assert not _platform("import numpy; import src; " + REPORT)["pinned"]
