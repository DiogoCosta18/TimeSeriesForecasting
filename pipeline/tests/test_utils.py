"""Helpers shared by the stages."""
from __future__ import annotations

import os

from src.utils import effective_cpu_count


def test_effective_cpu_count_respects_a_cgroup_v2_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(32)), raising=False)
    (tmp_path / "cpu.max").write_text("960000 100000\n", encoding="utf-8")
    assert effective_cpu_count(tmp_path) == 9          # 9.6 CPUs of quota: nine workers
    (tmp_path / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    assert effective_cpu_count(tmp_path) == 32
    (tmp_path / "cpu.max").write_text("50000 100000\n", encoding="utf-8")
    assert effective_cpu_count(tmp_path) == 1
    assert effective_cpu_count(tmp_path / "absent") == 32


def test_effective_cpu_count_respects_a_cgroup_v1_quota(tmp_path, monkeypatch):
    """As on the vast.ai pilot machine: 255 CPUs visible, a quota of 61.2."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(255)), raising=False)
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("6120000\n", encoding="utf-8")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n", encoding="utf-8")
    assert effective_cpu_count(tmp_path) == 61
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1\n", encoding="utf-8")
    assert effective_cpu_count(tmp_path) == 255
