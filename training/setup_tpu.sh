#!/bin/bash
set -e

cd ~/scot

if [ ! -d ".venv" ]; then
    echo "Initializing virtual environment..."
    sudo apt-get update -y
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
    python3.11 -m venv .venv
fi

source .venv/bin/activate


# JAX with TPU support
pip install --upgrade pip
pip install 'google-tunix[prod]'

# Extra deps
pip install wandb huggingface_hub gcsfs datasets evaluate tqdm jsonlines python-dotenv peft

# Verify JAX sees the TPUs
python -c "import jax; print('TPU devices:', jax.devices())"

echo "TPU setup complete natively."
