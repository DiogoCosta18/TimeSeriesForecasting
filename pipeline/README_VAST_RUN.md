# Vast.ai Large Training Run

This workflow rents a Vast.ai GPU instance, uploads the local forecasting repository, runs smoke first, runs the full large experiment only if smoke succeeds, and downloads outputs back to this computer with local `rsync` pulls over SSH.

The default sync mode is local download. The Vast instance is not expected to push files to your laptop.

## Required Local Tools

- Python 3
- `ssh`
- `rsync`
- Vast.ai API key in `VAST_API_KEY`

The launcher installs the `vastai` CLI if it is missing and runs:

```bash
vastai set api-key "$VAST_API_KEY"
```

## Local Output Directory

Use:

```bash
export OUTPUT_SYNC_URI="local:./vast_downloads/forecast_large_$(date +%Y%m%d_%H%M%S)"
```

If `OUTPUT_SYNC_URI` is empty, the launcher defaults to:

```bash
local:./vast_downloads/forecast_large_<timestamp>
```

Files are written locally under:

```text
<LOCAL_OUTPUT_DIR>/
  downloads/              # pulled /workspace/outputs from Vast
  orchestration_logs/     # local launch/upload/download/verify/destroy logs
  status_snapshots/       # copied STATUS.json snapshots
```

## Recommended First Large Run

Keep the instance after completion so you can inspect large model artifacts before manual cleanup:

```bash
export VAST_API_KEY="..."
export OUTPUT_SYNC_URI="local:./vast_downloads/forecast_large_$(date +%Y%m%d_%H%M%S)"
export VAST_AUTO_DESTROY=0
bash scripts/vast_launch_train_destroy.sh
```

If you run from the parent directory, use:

```bash
bash forecast_pipeline/scripts/vast_launch_train_destroy.sh
```

or, for this repository name:

```bash
bash forecasting_pipeline/scripts/vast_launch_train_destroy.sh
```

## Dry Run / Show Selected Offer

This searches and ranks offers, writes `selected_offer.json`, and does not create an instance:

```bash
export VAST_API_KEY="..."
bash scripts/vast_launch_train_destroy.sh --dry-run
```

## Optional Auto-Destroy

Auto-destroy is disabled by default. To destroy only after smoke succeeds, full large finishes, final local download completes, and local verification passes:

```bash
export VAST_API_KEY="..."
export OUTPUT_SYNC_URI="local:./vast_downloads/forecast_large_$(date +%Y%m%d_%H%M%S)"
export VAST_AUTO_DESTROY=1
bash scripts/vast_launch_train_destroy.sh
```

Never use auto-destroy for the first large run unless you are comfortable with the local verification gate and storage location.

## Optional Offer Controls

```bash
export VAST_MAX_PRICE="0.25"
export VAST_MIN_RELIABILITY="0.99"
export VAST_DISK_GB="500"
export VAST_AUTO_DESTROY="0"
```

The launcher searches for the best value GPU offer by default:

- highest `DLPerf / $/hr` under `VAST_MAX_PRICE`
- 1 GPU
- at least 23 GB GPU RAM
- verified host
- reliability around 99% or better, with progressive relaxation
- price preferably under `$VAST_MAX_PRICE`
- SSH enabled
- PyTorch CUDA image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`

To force a specific GPU family, set for example:

```bash
export VAST_GPU_NAME="RTX_3090"
```

Otherwise `VAST_GPU_NAME` is empty and the launcher may choose another 24 GB or larger GPU if it has better `DLPerf / $/hr`.

Selected offer and instance metadata are logged to:

```text
<LOCAL_OUTPUT_DIR>/orchestration_logs/selected_offer.json
<LOCAL_OUTPUT_DIR>/orchestration_logs/instance.json
<LOCAL_OUTPUT_DIR>/orchestration_logs/ssh_info.txt
```

## What Runs Remotely

Remote path:

```text
/workspace/forecast_pipeline
/workspace/outputs
```

The launcher uploads this repository, copies:

```text
/workspace/run_forecast_pipeline_vast.sh
/workspace/remote_heartbeat_status.sh
```

and starts the runner inside `tmux`, `screen`, or `nohup`.

The remote runner:

1. Installs system utilities where possible.
2. Creates `/workspace/venv`.
3. Installs `requirements.txt`.
4. Writes environment snapshots.
5. Starts a heartbeat at `/workspace/outputs/VAST_HEARTBEAT.log`.
6. Runs tests and smoke into `/workspace/outputs/smoke.log`.
7. Runs the large full experiment into `/workspace/outputs/full_large.log` only if smoke passes.

Large run command:

```bash
python -m src.main \
  --config configs/generated_large_everything.yaml \
  --monthly-config configs/monthly.yaml \
  --quarterly-config configs/quarterly.yaml \
  --vast-config configs/vast_gpu.yaml \
  --run-root /workspace/outputs \
  --budget large \
  --resume true \
  --finetuning-modes no_finetune,finetune_by_feature_bucket,finetune_by_series
```

NeuralForecast and Transformer auto configs use Optuna, not Ray.

## Monitoring

SSH command is written to:

```text
<LOCAL_OUTPUT_DIR>/orchestration_logs/ssh_info.txt
```

Typical command:

```bash
ssh -p <SSH_PORT> -o StrictHostKeyChecking=no root@<SSH_HOST>
```

Remote status:

```bash
tail -f /workspace/outputs/VAST_HEARTBEAT.log
tail -f /workspace/outputs/full_large.log
find /workspace/outputs -name STATUS.json -print
```

Local progress is pulled every 30 minutes by default. Override:

```bash
export DOWNLOAD_INTERVAL_SECONDS=900
```

The first long phase is CPU feature computation, so GPU usage can remain low before model training starts. To control CPU parallelism on Vast, set:

```bash
export FEATURE_COMPUTE_N_JOBS=$(($(nproc)-2))
```

If unset, the pipeline defaults to `max(1, os.cpu_count() - 2)`. During this stage, `STATUS.json` should show `current_stage="compute_features"` plus completed/total series counts.

## Manual Download

Download or resume downloads from an existing instance:

```bash
bash scripts/download_from_vast.sh <INSTANCE_ID> ./vast_downloads/manual_download
```

The helper downloads `/workspace/outputs/` to:

```text
./vast_downloads/manual_download/downloads/
```

It uses:

```bash
rsync -azP --partial --append-verify \
  -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10" \
  root@$SSH_HOST:/workspace/outputs/ \
  "$LOCAL_OUTPUT_DIR/downloads/"
```

No `--delete` is used by default. Set `LOCAL_RSYNC_DELETE=1` only if you explicitly want deletion.

## Resume Training

If your local terminal dies, the remote job should continue in `tmux`, `screen`, or `nohup`.

Resume local downloads:

```bash
bash scripts/download_from_vast.sh <INSTANCE_ID> <LOCAL_OUTPUT_DIR>
```

Resume training manually over SSH:

```bash
cd /workspace/forecast_pipeline
source /workspace/venv/bin/activate
python -m src.main \
  --config configs/generated_large_everything.yaml \
  --monthly-config configs/monthly.yaml \
  --quarterly-config configs/quarterly.yaml \
  --vast-config configs/vast_gpu.yaml \
  --run-root /workspace/outputs \
  --budget large \
  --resume true \
  --finetuning-modes no_finetune,finetune_by_feature_bucket,finetune_by_series
```

## Local Verification

Before destruction, verification is performed locally under:

```text
<LOCAL_OUTPUT_DIR>/downloads/
```

Required non-empty files:

- full run `DONE.json`
- full run `run_config.json`
- full run `data_manifest.json`
- full run `STATUS.json`
- full run `reports/leaderboard_overall.csv`
- full run `reports/summary.md`
- `smoke.log`
- `full_large.log`
- `LOCAL_DOWNLOAD_COMPLETE.json`

Required directories:

- full run `final/model_artifacts/`
- full run `checkpoints/`

If verification fails, the instance is never destroyed. Recovery commands are printed.

Verification and orchestration logs:

```text
<LOCAL_OUTPUT_DIR>/orchestration_logs/launch.log
<LOCAL_OUTPUT_DIR>/orchestration_logs/upload.log
<LOCAL_OUTPUT_DIR>/orchestration_logs/download.log
<LOCAL_OUTPUT_DIR>/orchestration_logs/download_verify.log
<LOCAL_OUTPUT_DIR>/orchestration_logs/destroy.log
```

## Manual Destroy

After inspection:

```bash
vastai destroy instance <INSTANCE_ID>
```

If `VAST_AUTO_DESTROY=0`, the script prints the manual download and destroy commands and preserves the instance.
