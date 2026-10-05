"""Helpers shared by the stages."""
from __future__ import annotations

import os

from src.utils import effective_cpu_count


def test_effective_cpu_count_respects_the_container_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(32)), raising=False)
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("960000 100000\n", encoding="utf-8")
    assert effective_cpu_count(cpu_max) == 9          # 9.6 CPUs of quota: nine workers
    cpu_max.write_text("max 100000\n", encoding="utf-8")
    assert effective_cpu_count(cpu_max) == 32
    assert effective_cpu_count(tmp_path / "absent") == 32
    cpu_max.write_text("50000 100000\n", encoding="utf-8")
    assert effective_cpu_count(cpu_max) == 1
