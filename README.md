# When Does Seasonal Decomposition Help?

Code of a controlled study of STL decomposition in time series forecasting: on 8,493 monthly and
quarterly series of the M3 and M4 competitions, twelve models from four families (statistical, machine
learning, neural, transformer) forecast each series directly and under two STL strategies, and every
series is described by six interpretable features. A second experiment tests whether global models
trained on feature-homogeneous groups of series beat a single global model.

## Repository

| Path | Content |
|---|---|
| `pipeline/src/` | The pipeline: data loading, features, sampling, tuning, evaluation, merge, gates and analysis. |
| `pipeline/configs/` | The run configuration (`rerun_v2.yaml`) and the pilot configuration. |
| `pipeline/environment/` | The pinned environment: Python 3.11.10 with an exact lockfile. |
| `pipeline/data_manifest/` | The manifest of the frozen M3/M4 copy the pipeline reads. |

See `pipeline/README.md` for the stages and how to run them.

## Reproducing the results

The results in the paper come from the run `full3_20261011` with `pipeline/configs/rerun_v2.yaml`, at
commit `e954eef` ("Tuning T3: neural trials scored with the ML studies' MASE (v2.8)"). The tag
`protocol-v1.4-approved` marks the code at the approval of the experimental protocol. The paper and the
protocol cite commits by their original hashes (`24b7274` for the run commit, `98e8368` for the data
freeze); the code at those commits is identical to the code at `e954eef` and `fb3f520`.

## Data

The M3 and M4 competition data are publicly available. The pipeline reads only the frozen copy described
by `pipeline/data_manifest/m3m4_v1.json`, which records the source and checksum of every file. Run
outputs (forecasts, frozen configurations and analysis tables) are not stored in this repository.
