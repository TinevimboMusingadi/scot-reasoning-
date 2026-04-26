#!/bin/bash
sudo apt-get update
sudo apt-get install -y unzip python3-venv
mkdir -p ~/scot
echo "Unzipping payload..."
unzip -o /tmp/scot_payload.zip -d ~/scot/
cd ~/scot

echo "Building Python environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing tunix and dependencies..."
pip install 'google-tunix[prod]'
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install wandb huggingface_hub gcsfs datasets evaluate tqdm jsonlines python-dotenv

echo "Python environment build complete."
python3 -c "import jax; print('TPU devices:', jax.devices())"

echo "Triggering background orchestrator..."
chmod +x training/run_all.sh
nohup bash training/run_all.sh > ~/scot/orchestrator.log 2>&1 &
echo "Background run initiated! You can safely disconnect."
