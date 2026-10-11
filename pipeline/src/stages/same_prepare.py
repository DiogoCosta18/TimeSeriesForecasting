"""Check that a new prepare reproduces a reference one (protocol change log v2.6 (d)).

    python -m src.stages.same_prepare REFERENCE NEW

REFERENCE and NEW are prepare directories (or their bundle.json). The bundle hash covers the
code commit and the configuration, so it changes with any commit or tuning budget even when the
prepared data are the same; this compares what prepare produced instead: every output file (its
SHA-256 and rows), the counts, the bucket summary, the frozen data files, the environment lock
and the configuration outside `tuning`. NEW's files are also checked on disk against its bundle.
Prints every difference and exits 1 if there is one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.data.frozen import sha256_file

SAME = ("counts", "bucket_summary", "data", "environment_lock_sha256")


def _bundle(path: Path) -> tuple[dict, Path]:
    path = Path(path)
    if path.is_dir():
        path = path / "bundle.json"
    return json.loads(path.read_text(encoding="utf-8")), path.parent


def differences(reference: Path, new: Path) -> list[str]:
    ref, _ = _bundle(reference)
    cur, folder = _bundle(new)
    out = []
    for name in sorted(ref["files"].keys() | cur["files"].keys()):
        a, b = ref["files"].get(name), cur["files"].get(name)
        if a is None or b is None:
            out.append(f"file {name}: only in {'new' if a is None else 'reference'}")
        elif (a["sha256"], a["rows"]) != (b["sha256"], b["rows"]):
            out.append(f"file {name}: sha256 {a['sha256'][:12]} -> {b['sha256'][:12]}, rows {a['rows']} -> {b['rows']}")
        elif sha256_file(folder / b["file"]) != b["sha256"]:
            out.append(f"file {name}: {b['file']} on disk does not match the new bundle")
    out += [f"{key} differs" for key in SAME if ref.get(key) != cur.get(key)]
    strip = lambda config: {k: v for k, v in config.items() if k != "tuning"}  # noqa: E731
    if strip(ref["config"]) != strip(cur["config"]):
        out.append("config outside tuning differs")
    return out


def main(argv: list[str] | None = None) -> int:
    reference, new = argv if argv is not None else sys.argv[1:]
    found = differences(Path(reference), Path(new))
    cpus = [_bundle(Path(p))[0].get("platform", {}).get("cpu_model") for p in (reference, new)]
    print(f"prepare check: reference {reference} (cpu {cpus[0]}), new {new} (cpu {cpus[1]})")
    for line in found:
        print(f"  DIFFERS {line}")
    print("prepare check: identical" if not found else f"prepare check: {len(found)} differences")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
