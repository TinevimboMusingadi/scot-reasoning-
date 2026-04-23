#!/bin/bash
# setup_tpu.sh
# Run ON the TPU to initialize dependencies

cd ~/scot

if [ ! -d ".venv" ]; then
    echo "Initializing virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# JAX with TPU support
pip install --upgrade pip
pip install 'google-tunix[prod]'

# Extra deps
pip install wandb huggingface_hub gcsfs datasets evaluate tqdm jsonlines python-dotenv peft

# Verify JAX sees the TPUs
python3 -c "import jax; print('TPU devices:', jax.devices())"

echo "TPU setup complete natively."
