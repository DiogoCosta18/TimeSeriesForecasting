#!/usr/bin/env bash
# Sensitivity analysis S2 (protocol change log v2.4) on one rented machine: the S1 script with
# STUDY=s2 (module src.stages.sensitivity2, logs RUN_DIR/logs/s2-*.log). The analysis runs where
# the run's merged rows are, with `python -m src.stages.sensitivity2 analyse`.
#
#   scripts/run_s2.sh CONFIG RUN_DIR DATA_DIR
set -euo pipefail
STUDY=s2 exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_s1.sh" "$@"
