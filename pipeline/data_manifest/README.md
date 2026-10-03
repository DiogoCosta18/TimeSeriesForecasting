# Frozen data copy (protocol D2, D20)

`m3m4_v1.json` identifies the frozen M3/M4 copy `m3m4-v1`: the SHA-256 of every raw
file (the original `M3C.xls` included) and of every frozen file, content hashes, series
counts and lengths, source provenance and the M3 cross-checks. It was written by
`python -m src.data.freeze_data freeze` at commit `98e8368` (WSL, pinned environment).
The pipeline reads the data only through this manifest (`src/data/frozen.py`): any
missing or altered file stops the run.

## Where the copy is

A private Backblaze B2 bucket (server-side encryption on), path
`data/frozen_m3m4_v1/`: 19 files, 235 MB, uploaded on 3 October 2026 and verified
three ways: rclone size and SHA-1 check, a byte-for-byte comparison after download,
and `freeze_data verify` on a fresh download against this manifest.

Access is through an rclone remote named `b2rerun`, created by D. Costa with an
application key restricted to that bucket. Keys and the bucket name are not stored in
this repository.

## Getting the data on a machine

```bash
export RERUN_BUCKET=<bucket name>                  # set on the machine, not in the repo
rclone copy "b2rerun:$RERUN_BUCKET/data/frozen_m3m4_v1" /workspace/data/frozen_m3m4_v1 --checksum
python -m src.data.freeze_data verify --data-dir /workspace/data/frozen_m3m4_v1
export RERUN_DATA_DIR=/workspace/data/frozen_m3m4_v1   # read by the pipeline's loader
```

`verify` must print "All frozen files verified." before anything else runs.

## Re-creating the copy

Only to audit it: check out commit `98e8368` and run, with the original `M3C.xls`
downloaded in a browser from the International Institute of Forecasters (the site
refuses scripted downloads; the file is refused unless its SHA-256 is `23bfbba2...`):

```bash
python -m src.data.freeze_data freeze --out NEW_EMPTY_DIR \
    --code-commit 98e83684c40dc9e86301b082eeb6c57c17a98081 --m3c PATH/M3C.xls
```

The frozen files' content hashes must equal those in `m3m4_v1.json`.
