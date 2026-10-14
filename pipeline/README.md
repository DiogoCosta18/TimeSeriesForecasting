# Rerun v2 pipeline

This pipeline implements the experimental protocol of the study of STL decomposition and
feature-specialised training on M3/M4. The decisions (D1–D30) and gates (G1–G13) cited in the code
refer to that protocol.

Nothing in the pipeline substitutes a result. Every forecast comes from a trained model, and a
failure is recorded as a failure, never replaced. Every output carries the code commit, the
environment lock, the data manifest, the prepare bundle and the frozen configurations it came from.

## Environment

Python 3.11.10 with the pinned lock in `environment/` (see `environment/README.md`). Stages refuse
to run from a working tree with uncommitted changes. On a machine without git metadata,
`RERUN_CODE_COMMIT` must hold the full 40-character commit.

## Data

The frozen M3/M4 copy `m3m4-v1` is described by `data_manifest/m3m4_v1.json`. Fetching and
verifying it is described in `data_manifest/README.md`. The loader reads nothing else.

## Stages

```bash
python -m src.cli freeze-data verify --data-dir DATA            # the frozen copy is intact
python -m src.cli prepare  --config configs/rerun_v2.yaml --run RUN --data-dir DATA [--jobs N|auto]
python -m src.cli tune     --config configs/rerun_v2.yaml --run RUN --data-dir DATA --shard FAMILY-FREQUENCY
python -m src.cli freeze   --config configs/rerun_v2.yaml --run RUN
python -m src.cli evaluate --config configs/rerun_v2.yaml --run RUN --data-dir DATA --shard FAMILY-FREQUENCY
python -m src.cli merge    --run RUN
python -m src.cli gates    --run RUN                             # exit code 1 if any gate fails
python -m src.cli analyse  --run RUN --out OUT
```

Shards are family–frequency pairs: `statistical-`, `ml-`, `neural-` and `transformer-` with
`monthly` or `quarterly`. Tuning has no statistical shards.

| Stage | Writes (under `RUN/`) | Refuses |
|---|---|---|
| prepare (R1) | `prepare/`: eligibility, features, samples, buckets, tuning set, cutoffs, `bundle.json` | an existing bundle; a resume with other inputs |
| tune (R2) | `tune/<shard>/`: one entry per study, `studies.sqlite` | a configuration, manifest or prepare file that differs from the bundle's |
| freeze (R2b) | `configs_frozen.json` with its hash | a missing study; studies with different provenance; an existing file |
| evaluate (R3) | `evaluate/<shard>/tasks/`: one parquet and completion record per task | any provenance other than the frozen configurations' (see D16 below) |
| merge (R4) | `merged/`: rows, failed rows, `merge.json` with a result hash | changed or unknown outputs; mismatched hashes; unpaired tercile rows |
| gates | `merged/gates.json` | — (reports G1–G13) |
| analyse (R5) | every table and figure of the analysis, `analysis.json` | failed or stale gates; a non-empty output folder |

Interrupted stages resume. A finished task is skipped only if its record and output still match
the current provenance; a task produced under other provenance stops the stage instead of being
mixed in.

**Failures (D16).** A statistical model's failure on a series is a failed row; the task goes on.
A global task that raises is retried once. If it fails again it is listed in
`evaluate/<shard>/failures/` and the stage ends with an error after the remaining tasks. After the
cause is fixed and committed, only the listed task is rerun:

```bash
python -m src.cli evaluate ... --shard SHARD --only TASK_ID
```

The fix must descend from the frozen commit, and the environment, data, bundle and configurations
must be unchanged. The rerun is entered in `evaluate/<shard>/d16_reruns.json`, the only way the
merge accepts a second commit (G4).

## Pilot

The pilot runs the same code with `configs/pilot_v2.yaml`: two features (evolving seasonality and
nonlinearity), 100 series per source per feature in 25 strata, a tuning set of 100 per source,
3 trials per study, main seed only. Nothing else differs from `configs/rerun_v2.yaml`. Pilot results
are never reported.
