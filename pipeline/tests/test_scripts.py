"""D20 scripts against a local stand-in for the bucket (rclone's local backend).

Skipped where bash or rclone is missing; the run machines have both.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RCLONE = os.environ.get("RCLONE") or shutil.which("rclone") or str(Path.home() / ".local/bin/rclone")
pytestmark = pytest.mark.skipif(not (shutil.which("bash") and Path(RCLONE).exists() and os.name == "posix"),
                                reason="needs bash and rclone")


@pytest.fixture
def env(tmp_path):
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    return {**os.environ, "RCLONE": RCLONE, "RERUN_REMOTE": "standin", "RCLONE_CONFIG_STANDIN_TYPE": "local",
            "RERUN_BUCKET": str(bucket)}, bucket


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_sync_uploads_without_temporaries_never_deletes_and_verifies_at_the_end(tmp_path, env):
    environ, bucket = env
    run = tmp_path / "pilot_run"
    (run / "prepare").mkdir(parents=True)
    (run / "prepare" / "bundle.json").write_text("{}", encoding="utf-8")
    (run / "prepare" / ".samples.parquet.tmp").write_text("partial", encoding="utf-8")
    sync = subprocess.Popen(["bash", str(SCRIPTS / "sync_run.sh"), str(run), "1"], env=environ,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    remote = bucket / "runs" / "pilot_run"
    for _ in range(60):
        if (remote / "prepare" / "bundle.json").exists():
            break
        time.sleep(0.5)
    (run / "prepare" / "bundle.json").unlink()                       # a local loss ...
    (run / "evaluate" / "ml-monthly").mkdir(parents=True)
    (run / "evaluate" / "ml-monthly" / "heartbeat.json").write_text('{"done": 3}', encoding="utf-8")
    (run / ".sync_stop").touch()
    out, _ = sync.communicate(timeout=120)
    assert sync.returncode == 0, out
    assert "final sync verified" in out
    assert _files(remote) == {"prepare/bundle.json": b"{}", "evaluate/ml-monthly/heartbeat.json": b'{"done": 3}'}  # ... is not propagated


def test_sync_fails_without_a_bucket_name(tmp_path, env):
    environ, _ = env
    environ = {k: v for k, v in environ.items() if k != "RERUN_BUCKET"}
    res = subprocess.run(["bash", str(SCRIPTS / "sync_run.sh"), str(tmp_path), "1"], env=environ, capture_output=True, text=True)
    assert res.returncode != 0 and "RERUN_BUCKET" in res.stderr


def test_fetch_gets_the_requested_parts_and_checks_them(tmp_path, env):
    environ, bucket = env
    remote = bucket / "runs" / "v2"
    for rel, text in {"prepare/bundle.json": "b", "prepare/cutoffs.parquet": "c", "configs_frozen.json": "f",
                      "tune/ml-monthly/tasks/x.json": "t", "evaluate/ml-monthly/tasks/y.parquet": "e"}.items():
        (remote / rel).parent.mkdir(parents=True, exist_ok=True)
        (remote / rel).write_text(text, encoding="utf-8")
    dest = tmp_path / "machine" / "v2"
    res = subprocess.run(["bash", str(SCRIPTS / "fetch_run.sh"), "v2", str(dest), "prepare", "configs_frozen.json"],
                         env=environ, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert _files(dest) == {"prepare/bundle.json": b"b", "prepare/cutoffs.parquet": b"c", "configs_frozen.json": b"f"}
    res = subprocess.run(["bash", str(SCRIPTS / "fetch_run.sh"), "v2", str(tmp_path / "all")], env=environ, capture_output=True, text=True)
    assert res.returncode == 0 and len(_files(tmp_path / "all")) == 5


def test_run_stage_logs_into_the_run_and_keeps_the_exit_code(tmp_path):
    run = tmp_path / "run"
    environ = {**os.environ, "PYTHON": shutil.which("python") or "python3"}
    res = subprocess.run(["bash", str(SCRIPTS / "run_stage.sh"), str(run), "gates", "gates", "--run", str(run)],
                         env=environ, capture_output=True, text=True, cwd=SCRIPTS.parent)
    assert res.returncode != 0                                      # no merged bundle: the stage fails ...
    log = (run / "logs" / "gates.log").read_text(encoding="utf-8")
    assert "python -m src.cli gates --run" in log and "exit 1" in log  # ... and the log says so
