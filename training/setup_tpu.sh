#!/bin/bash
# Usage: bash training/setup_tpu.sh
# Run from your LOCAL machine after TPU state = ACTIVE

TPU_NAME="my-tpu-node"
ZONE="us-east5-a"
PROJECT="tpu-builder1"

gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE \
    --project=$PROJECT \
    --command="
        mkdir -p ~/scot
        cd ~/scot
        # Python environment
        python3 -m venv .venv
        source .venv/bin/activate

        # JAX with TPU support
        pip install --upgrade pip
        pip install 'google-tunix[prod]'

        # Extra deps
        pip install wandb huggingface_hub gcsfs datasets evaluate tqdm jsonlines python-dotenv

        # Verify JAX sees the TPUs
        python3 -c \"import jax; print('TPU devices:', jax.devices())\"
    "

echo "TPU setup complete."
