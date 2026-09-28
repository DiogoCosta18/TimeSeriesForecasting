#!/usr/bin/env bash
# Build the pinned rerun environment on Linux x86_64 (vast.ai machines, or WSL
# Ubuntu on a Windows laptop) from the committed lockfile, then verify it.
#
# Usage:  bash pipeline/environment/build_env.sh [VENV_DIR]
#         (default VENV_DIR: $HOME/venvs/rerun-v2)
#
# Steps: install the pinned uv release if needed; install Python 3.11.10;
# create a fresh virtual environment; install exactly the lockfile; run the
# environment tests (lockfile match and API contract, protocol test I5).
# The script exits non-zero if any step or test fails.
set -euo pipefail

UV_VERSION="0.9.30"
PYTHON_VERSION="3.11.10"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$HERE")"
LOCK="$HERE/requirements-lock-linux-cu121.txt"
VENV="${1:-$HOME/venvs/rerun-v2}"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1 || [[ "$(uv --version | awk '{print $2}')" != "$UV_VERSION" ]]; then
    echo "Installing uv $UV_VERSION into $HOME/.local/bin"
    mkdir -p "$HOME/.local/bin"
    curl -LsSf "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
        | tar xz -C "$HOME/.local/bin" --strip-components=1
fi

uv python install "$PYTHON_VERSION"

# Recreate the environment from scratch, but only ever delete a directory that
# is recognisably a virtual environment.
if [[ -e "$VENV" ]]; then
    if [[ -f "$VENV/pyvenv.cfg" ]]; then
        rm -rf "$VENV"
    else
        echo "Refusing to delete $VENV: it is not a virtual environment." >&2
        exit 1
    fi
fi
uv venv --python "$PYTHON_VERSION" "$VENV"
uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match -r "$LOCK"

cd "$PIPELINE_DIR"
"$VENV/bin/python" -m pytest tests/test_env_lock.py tests/test_env_api_contract.py
echo "Environment OK: $VENV (lockfile sha256 $(sha256sum "$LOCK" | cut -c1-16))"
