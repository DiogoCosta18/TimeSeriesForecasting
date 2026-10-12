# When Does Seasonal Decomposition Help?

Code and protocol of a controlled study of STL decomposition in time series forecasting: on 8,493
monthly and quarterly series of the M3 and M4 competitions, twelve models from four families
(statistical, machine learning, neural, transformer) forecast each series directly and under two STL
strategies, and every series is described by six interpretable features. A second experiment tests
whether global models trained on feature-homogeneous groups of series beat a single global model.

## Repository

| Path | Content |
|---|---|
| `pipeline/` | The pipeline: data loading, features, sampling, tuning, evaluation, merge, gates and analysis (`src/`), configurations (`configs/`), tests (`tests/`), machine scripts (`scripts/`), the pinned environment (`environment/`) and the data manifest (`data_manifest/`). See `pipeline/README.md`. |
| `pipeline/docs/protocol/` | The experimental protocol (`rerun_protocol.pdf`): every decision, test and gate fixed before the run, with a change log of every later amendment and of each run. The tag `protocol-v1.4-approved` marks the approved version. |
| `pipeline/docs/provenance/` | The commit map described below. |

## Reproducing the results

`pipeline/README.md` describes the environment (Python 3.11.10 with a pinned lockfile), the frozen
M3/M4 copy and the stages (prepare, tune, freeze, evaluate, merge, gates, analyse). The results in the
paper come from the run `full3_20261011` with `pipeline/configs/rerun_v2.yaml`, at the code of original
commit `24b7274` (published commit `7f527d4`); the protocol's change log, versions 2.6 to 2.9, records
that run, its checks and its results.

## Commit hashes

The published history differs from the original history only in its commit messages; the code of every
commit is unchanged. The protocol, the paper and the run records cite original commit hashes;
`pipeline/docs/provenance/commit_map.tsv` maps each of them to its published commit.

## Data

The M3 and M4 competition data are publicly available. The pipeline reads only the frozen copy described
by `pipeline/data_manifest/m3m4_v1.json`, which records the source and checksum of every file. Run
outputs (forecasts, frozen configurations and analysis tables) are not stored in this repository.
