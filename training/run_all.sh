#!/bin/bash
# run_all.sh
# To be executed ON the TPU node using watchdog over SSH

cd ~/scot
source .venv/bin/activate

# Ensure bucket exists
gcloud storage buckets create gs://tpu-builder1-scot-checkpoints --project tpu-builder1 --location=us-east5 || true

echo "======================================"
echo "Synchronizing Previous Checkpoints from GCS"
echo "======================================"
mkdir -p ~/scot/outputs
gcloud storage cp -r gs://tpu-builder1-scot-checkpoints/* ~/scot/outputs/ || true

echo "======================================"
echo "Starting Background Checkpoint Sync Daemon"
echo "======================================"
# This loop will push outputs to GCS iteratively every 5 minutes to guarantee survival against 4hr preemptions
(
  while true; do
    sleep 300
    echo "[Daemon] Synchronizing checkpoints to GCS..."
    gcloud storage rsync -r ~/scot/outputs gs://tpu-builder1-scot-checkpoints/ > /dev/null 2>&1 || true
  done
) &
SYNC_PID=$!

echo "======================================"
echo "Starting S-CoT SFT Run for Qwen2.5-3B"
echo "======================================"
# No log redirection! Feed the matrix straight out to stdout for watchdog parsing!
python training/sft_scot.py
echo "S-CoT Run Finished."

echo "======================================"
echo "Starting Flat SFT Run for Qwen2.5-3B"
echo "======================================"
python training/sft_flat.py
echo "Flat Run Finished."

echo "All training scripts completed."

echo "======================================"
echo "Running S-CoT Inference on TPU"
echo "======================================"
python training/run_inference.py --model scot || echo "S-CoT inference failed (non-fatal)"

echo "======================================"
echo "Running Flat Inference on TPU"
echo "======================================"
python training/run_inference.py --model flat || echo "Flat inference failed (non-fatal)"

echo "All training and inference completed."

echo "======================================"
echo "Safeguarding Final Checkpoints to GCS"
echo "======================================"
# Terminate the backup loop cleanly
kill -9 $SYNC_PID || true
gcloud storage rsync -r ~/scot/outputs gs://tpu-builder1-scot-checkpoints/
echo "Saved successfully! You can download it directly from Google Cloud Storage or Colab using: gsutil cp -r gs://tpu-builder1-scot-checkpoints/ ./"
