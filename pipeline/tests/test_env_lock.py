"""The running environment must match the committed lockfile exactly (protocol D22, gate G4).

Every package pinned in environment/requirements-lock-linux-cu121.txt must be
installed at exactly that version, and the interpreter must be Python 3.11.10.
Run results store the lockfile hash; this test guarantees the hash describes the
environment that actually ran.
"""
from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "environment" / "requirements-lock-linux-cu121.txt"
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s#;]+)", re.MULTILINE)


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def test_python_version():
    assert sys.version_info[:3] == (3, 11, 10), f"expected Python 3.11.10, got {sys.version.split()[0]}"


def test_every_locked_package_installed_at_locked_version():
    pins = {_norm(n): v for n, v in PIN.findall(LOCK.read_text(encoding="utf-8"))}
    assert len(pins) > 100, "lockfile looks truncated"
    installed = {_norm(d.metadata["Name"]): d.version for d in metadata.distributions()}
    missing = sorted(n for n in pins if n not in installed)
    wrong = {n: (pins[n], installed[n]) for n in pins if n in installed and installed[n] != pins[n]}
    assert not missing, f"locked but not installed: {missing}"
    assert not wrong, f"installed version differs from lock (expected, found): {wrong}"
