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


# --- run_machine.sh: several machines coordinated through the bucket ------------------------------

FAKE_CLI = '''
import os, sys, time
from pathlib import Path

args = sys.argv[3:]                      # python -m src.cli STAGE ...
stage, opts = args[0], dict(zip(args[1::2], args[2::2]))
run = Path(opts["--run"])
shard = opts.get("--shard", "")
with open(os.environ["CALLS"], "a") as f:
    f.write(f"{time.time_ns()} {os.environ['MACHINE']} {stage} {shard} {opts.get('--workers', '')}\\n")
TUNE = ["ml-monthly", "ml-quarterly", "neural-monthly", "neural-quarterly", "transformer-monthly", "transformer-quarterly"]
EVAL = ["statistical-monthly", "statistical-quarterly"] + TUNE
need = {"tune": ["prepare/bundle.json"], "freeze": [f"tune/{s}/tasks/x.json" for s in TUNE],
        "evaluate": ["prepare/bundle.json", "configs_frozen.json"], "merge": [f"evaluate/{s}/tasks/x.parquet" for s in EVAL],
        "gates": ["merged/rows.parquet"], "analyse": ["merged/gates.json"]}
missing = [p for p in need.get(stage, []) if not (run / p).exists()]
if missing:
    sys.exit(f"{stage}: missing inputs {missing}")
if os.environ.get("FAIL") == f"{stage}-{shard}":
    sys.exit(f"{stage} {shard}: boom")
out = {"prepare": "prepare/bundle.json", "tune": f"tune/{shard}/tasks/x.json", "freeze": "configs_frozen.json",
       "evaluate": f"evaluate/{shard}/tasks/x.parquet", "merge": "merged/rows.parquet", "gates": "merged/gates.json",
       "analyse": "analysis/analysis.json"}[stage]
time.sleep(0.2)
(run / out).parent.mkdir(parents=True, exist_ok=True)
(run / out).write_text(f"{stage} {shard}")
if stage == "evaluate":                  # a temporary left by an interrupted atomic write
    (run / out).with_name(".y.parquet.tmp").write_text("partial")
'''

MACHINES = {  # the full run's allocation in miniature: the lead tunes and evaluates the CPU shards
    "L": {"LEAD": "1", "TUNE": "ml-monthly:3 ml-quarterly:3",
          "EVALUATE": "ml-monthly:2 ml-quarterly:2 | statistical-monthly:4 statistical-quarterly:4"},
    "A": {"TUNE": "neural-monthly neural-quarterly", "EVALUATE": "neural-monthly neural-quarterly"},
    "B": {"TUNE": "transformer-monthly transformer-quarterly", "EVALUATE": "transformer-quarterly | transformer-monthly"},
}


def _machines(tmp_path, env, fail=None):
    environ, bucket = env
    fake = tmp_path / "fake_cli.py"
    fake.write_text(FAKE_CLI, encoding="utf-8")
    python = tmp_path / "python"
    python.write_text(f"#!/bin/bash\nexec {shutil.which('python3') or 'python3'} {fake} \"$@\"\n", encoding="utf-8")
    python.chmod(0o755)
    calls = tmp_path / "calls.txt"
    procs = {}
    for name, extra in MACHINES.items():
        machine_env = {**environ, **extra, "MACHINE": name, "PYTHON": str(python), "CALLS": str(calls), "POLL": "1",
                       "RETRY_SLEEP": "0", "ON_DONE": f"touch {tmp_path / ('done_' + name)}",
                       **({"FAIL": fail} if fail and name == "B" else {})}
        run = tmp_path / f"machine_{name}" / "full_run"
        procs[name] = subprocess.Popen(["bash", str(SCRIPTS / "run_machine.sh"), "configs/rerun_v2.yaml", str(run), str(tmp_path / "data")],
                                       env=machine_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=SCRIPTS.parent)
    out = {name: p.communicate(timeout=300)[0] for name, p in procs.items()}
    codes = {name: p.returncode for name, p in procs.items()}
    rows = [line.split(" ") for line in calls.read_text().splitlines()]
    return codes, out, [(int(t), m, stage, shard, workers) for t, m, stage, shard, workers in rows], bucket / "runs" / "full_run"


def test_machines_run_the_stages_in_order_through_the_bucket(tmp_path, env):
    codes, out, calls, remote = _machines(tmp_path, env)
    assert codes == {"L": 0, "A": 0, "B": 0}, out
    first = {}
    last = {}
    for t, _, stage, *_ in calls:
        first.setdefault(stage, t)
        last[stage] = t
    assert [stage for _, _, stage, *_ in calls].count("prepare") == 1 and calls[0][1:3] == ("L", "prepare")
    assert last["prepare"] < first["tune"] and last["tune"] < first["freeze"] < first["evaluate"]
    assert last["evaluate"] < first["merge"] < first["gates"] < first["analyse"]
    assert {(m, stage, shard) for _, m, stage, shard, _ in calls if stage in ("tune", "evaluate")} == {
        (m, stage, entry.split(":")[0]) for m, spec in MACHINES.items() for stage in ("tune", "evaluate")
        for entry in spec[stage.upper()].replace("|", " ").split()}
    assert {(stage, shard, w) for _, _, stage, shard, w in calls if shard.startswith(("ml", "statistical"))} == {
        ("tune", "ml-monthly", "3"), ("tune", "ml-quarterly", "3"), ("evaluate", "ml-monthly", "2"),
        ("evaluate", "ml-quarterly", "2"), ("evaluate", "statistical-monthly", "4"), ("evaluate", "statistical-quarterly", "4")}
    assert all((tmp_path / f"done_{m}").exists() for m in MACHINES)          # ON_DONE after success
    assert (remote / "analysis" / "analysis.json").exists() and (remote / "coordination" / "analyse.done.json").exists()
    assert not list(remote.rglob("*.tmp"))                                    # temporaries never uploaded
    assert not any("--filter is recommended" in o for o in out.values())      # ordered filter rules only
    for m in ("A", "B"):                                                      # what the others fetched
        run = tmp_path / f"machine_{m}" / "full_run"
        assert (run / "prepare" / "bundle.json").exists() and (run / "configs_frozen.json").exists()
        assert not (run / "merged").exists()


def test_a_failed_shard_stops_the_lead_before_the_merge_and_keeps_machines_up(tmp_path, env):
    codes, out, calls, remote = _machines(tmp_path, env, fail="evaluate-transformer-quarterly")
    assert codes["B"] != 0 and codes["L"] != 0 and codes["A"] == 0, out
    assert "failed elsewhere: evaluate-transformer-quarterly" in out["L"]
    assert not any(stage in ("merge", "gates", "analyse") for _, _, stage, *_ in calls)
    assert (remote / "coordination" / "evaluate-transformer-quarterly.failed.json").exists()
    assert (remote / "logs" / "evaluate-transformer-quarterly.log").exists()
    assert not (tmp_path / "done_B").exists() and not (tmp_path / "done_L").exists() and (tmp_path / "done_A").exists()


def test_run_machine_knows_every_shard_of_the_protocol():
    import re

    from src.stages.tasks import tuning_tasks
    from src.forecast.registry import FAMILIES

    text = (SCRIPTS / "run_machine.sh").read_text(encoding="utf-8")
    tune = re.search(r'^TUNE_SHARDS_ALL="([^"]+)"', text, re.M).group(1).split()
    evaluate = re.search(r'^EVALUATE_SHARDS_ALL="([^"]+)"', text, re.M).group(1).replace("$TUNE_SHARDS_ALL", " ".join(tune)).split()
    assert sorted(tune) == sorted({t.shard for t in tuning_tasks()})
    assert sorted(evaluate) == sorted(f"{f}-{q}" for f in FAMILIES for q in ("monthly", "quarterly"))


# --- run_s1.sh: sensitivity analysis S1 on one machine ----------------------------------------------

FAKE_S1 = '''
import os, sys, time
args = sys.argv[3:]                      # python -m src.stages.sensitivity STAGE ...
stage, opts = args[0], dict(zip(args[1::2], args[2::2]))
with open(os.environ["CALLS"], "a") as f:
    f.write(f"{time.time_ns()} {stage} {opts.get('--group', '-')} {opts.get('--workers', '-')}" + chr(10))
time.sleep(0.2)
if os.environ.get("FAIL") == f"{stage}-{opts.get('--group')}":
    sys.exit("boom")
'''


def _run_s1(tmp_path, env, fail=None):
    environ, bucket = env
    fake = tmp_path / "fake_s1.py"
    fake.write_text(FAKE_S1, encoding="utf-8")
    python = tmp_path / "python"
    python.write_text(f"#!/bin/bash\nexec {shutil.which('python3') or 'python3'} {fake} \"$@\"\n", encoding="utf-8")
    python.chmod(0o755)
    calls = tmp_path / "calls.txt"
    run = tmp_path / "full_run"
    run.mkdir()
    machine_env = {**environ, "PYTHON": str(python), "CALLS": str(calls), "ON_DONE": f"touch {tmp_path / 'done'}",
                   **({"FAIL": fail} if fail else {})}
    res = subprocess.run(["bash", str(SCRIPTS / "run_s1.sh"), "configs/rerun_v2.yaml", str(run), str(tmp_path / "data")],
                         env=machine_env, capture_output=True, text=True, cwd=SCRIPTS.parent, timeout=300)
    rows = [line.split(" ") for line in calls.read_text().splitlines()] if calls.exists() else []
    return res, [(int(t), stage, group, workers) for t, stage, group, workers in rows], run, bucket / "runs" / "full_run"


def test_run_s1_orders_the_stages_and_stops_the_machine_after_success(tmp_path, env):
    res, calls, run, remote = _run_s1(tmp_path, env)
    assert res.returncode == 0, res.stdout + res.stderr
    stages = {(stage, group, workers) for _, stage, group, workers in calls}
    assert stages == {("tune", "cpu", "-"), ("tune", "gpu", "-"), ("freeze", "-", "-"), ("evaluate", "cpu", "8"), ("evaluate", "gpu", "-")}
    time_of = {(s, g): t for t, s, g, _ in calls}
    assert max(time_of[("tune", "cpu")], time_of[("tune", "gpu")]) < time_of[("freeze", "-")]
    assert time_of[("freeze", "-")] < min(time_of[("evaluate", "cpu")], time_of[("evaluate", "gpu")])
    assert (tmp_path / "done").exists() and "last upload checked" in res.stdout
    assert (remote / "logs" / "s1-evaluate-gpu.log").exists()               # uploaded with the run


def test_run_s1_failure_keeps_the_machine_up(tmp_path, env):
    res, calls, run, _ = _run_s1(tmp_path, env, fail="evaluate-gpu")
    assert res.returncode != 0 and not (tmp_path / "done").exists()
    assert "exit 1" in (run / "logs" / "s1-evaluate-gpu.log").read_text()
