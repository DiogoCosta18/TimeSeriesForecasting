# Pinned environment (rerun protocol D22)

Every experiment in the rerun runs in one environment: **Python 3.11.10 on Linux
x86_64**, with the exact package versions in `requirements-lock-linux-cu121.txt`.
Results record the SHA-256 of that lockfile, and `tests/test_env_lock.py` fails if
the running environment differs from it in any package.

## Files

| File | Role |
|---|---|
| `requirements-pinned.in` | Human-edited input: every direct and indirect dependency at an exact version, with the reason for each deviation from the source. |
| `requirements-lock-linux-cu121.txt` | Resolver output for the run platform (Linux, glibc 2.35+, Python 3.11, CUDA 12.1 build of torch). **This is what gets installed.** |
| `build_env.sh` | Builds the environment from the lock and runs the environment tests. Used on vast.ai and in WSL. |

## Where the versions come from

- `pip_freeze.txt` of the vast.ai machine of 21 May 2026 (the only full freeze).
  Its versions match the `environment.json` of the 23 May runs for every recorded
  package **except torch**.
- torch is `2.5.1+cu121`, the version recorded by the 23 May pilot and large
  runs. The 25 May and 28-30 May runs did not record their environment; their
  result rows record mlforecast 1.0.31, neuralforecast 3.1.8 and statsforecast
  2.0.3, which match.
- `sympy` is `1.13.1`, not the freeze's `1.14.0`: torch 2.5.1 requires exactly
  1.13.1 (the freeze's 1.14.0 came with torch 2.12).
- The CUDA 12.1 libraries and `triton 3.1.0` are the exact versions torch
  2.5.1+cu121 declares; the resolver adds them.
- `rdata 0.11.2` and `xarray 2026.7.0` are not from the May machine: they were
  added on 29 Sep 2026 for data preparation only. rdata reads the M3 data of
  Mcomp 2.7, which the freeze compares with the original `M3C.xls` (read with
  xlrd 2.0.2, already in the lock); protocol D2, amended in v1.3 and v1.4.
  Adding them changed no other pin.

## Build the environment

On any Linux x86_64 machine (vast.ai) or in WSL Ubuntu:

```bash
bash pipeline/environment/build_env.sh            # default: ~/venvs/rerun-v2
```

The script installs the pinned `uv` (0.9.30) if needed, installs Python 3.11.10,
creates a fresh virtual environment, installs the lock, and runs
`tests/test_env_lock.py` and `tests/test_env_api_contract.py` (protocol test I5).
It stops with a non-zero exit code if anything fails.

From Windows, run it inside WSL with Git Bash path conversion disabled:

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -- bash /mnt/c/<path-to-repo>/pipeline/environment/build_env.sh
```

**Native Windows cannot reproduce this environment.** On Windows with Python
below 3.12, `statsforecast 2.0.3` requires `numba < 0.60`, which in turn forces
numpy 1.x, while the lock (Linux) has numba 0.65.1 and numpy 2.4.6. Any local
development and testing therefore happens in WSL, in the same environment the
runs use.

## Regenerate the lock

Only after editing `requirements-pinned.in`, and in a commit of its own:

```bash
uv pip compile requirements-pinned.in \
    --python-version 3.11 --python-platform x86_64-manylinux_2_35 \
    --index-strategy unsafe-best-match --emit-index-url --no-header \
    --output-file requirements-lock-linux-cu121.txt
```

`--emit-index-url` keeps the PyTorch index line in the lock; without it
`torch==2.5.1+cu121` cannot be installed, because it is not on PyPI.
