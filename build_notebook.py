import json

notebook = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# S-CoT Distilled Model — Inference Notebook\\n",
        "This notebook downloads your TPU-trained LoRA checkpoints from GCS, injects them into the Qwen2.5-3B-Instruct base model, and runs inference.\\n",
        "\\n",
        "**Runtime:** Select T4 or L4 GPU in Colab.\\n",
        "\\n",
        "**Two models available:**\\n",
        "- `sft-scot`: S-CoT structured reasoning (loss=1.27, ppl=3.55)\\n",
        "- `sft-flat`: Flat baseline (loss=0.275, ppl=1.32)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Install dependencies — pin numpy<2 to prevent C-extension ABI mismatch\\n",
        "!pip install -q 'numpy<2.0.0'\\n",
        "!pip install -q 'google-tunix[prod]'\\n",
        "!pip install -q -U 'transformers>=4.45,<=4.57.1'\\n",
        "!pip install -q wandb huggingface_hub gcsfs datasets evaluate tqdm peft\\n",
        "\\n",
        "# Restart runtime so numpy C-extensions reload cleanly\\n",
        "import os\\n",
        "os.kill(os.getpid(), 9)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Authenticate and download checkpoints\\n",
        "from google.colab import auth\\n",
        "auth.authenticate_user()\\n",
        "\\n",
        "# Download both S-CoT and Flat checkpoints from GCS\\n",
        "!mkdir -p /content/checkpoints/sft-scot\\n",
        "!mkdir -p /content/checkpoints/sft-flat\\n",
        "!gsutil -m cp -r gs://tpu-builder1-scot-checkpoints/sft-scot/* /content/checkpoints/sft-scot/ 2>/dev/null || echo 'S-CoT checkpoint not found in bucket'\\n",
        "!gsutil -m cp -r gs://tpu-builder1-scot-checkpoints/sft-flat/* /content/checkpoints/sft-flat/ 2>/dev/null || echo 'Flat checkpoint not found in bucket'\\n",
        "\\n",
        "# Show what we got\\n",
        "!echo '=== S-CoT ===' && ls /content/checkpoints/sft-scot/ 2>/dev/null || echo '(empty)'\\n",
        "!echo '=== Flat ===' && ls /content/checkpoints/sft-flat/ 2>/dev/null || echo '(empty)'"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "#@title Select Model to Load { run: 'auto' }\\n",
        "MODEL_VARIANT = 'sft-scot' #@param ['sft-scot', 'sft-flat']\\n",
        "\\n",
        "import jax\\n",
        "import jax.numpy as jnp\\n",
        "from transformers import AutoTokenizer\\n",
        "from tunix.models.qwen2 import model as qwen_model\\n",
        "from tunix.models.qwen2 import params as qwen_params\\n",
        "from qwix._src.providers import lora as qwix_lora\\n",
        "from flax import nnx\\n",
        "from orbax import checkpoint as ocp\\n",
        "from huggingface_hub import snapshot_download\\n",
        "import os\\n",
        "\\n",
        "base_model_id = 'Qwen/Qwen2.5-3B-Instruct'\\n",
        "tokenizer = AutoTokenizer.from_pretrained(base_model_id)\\n",
        "\\n",
        "# Build base model from HuggingFace weights\\n",
        "print('Building base model...')\\n",
        "config = qwen_model.ModelConfig.qwen2p5_3b()\\n",
        "n = len(jax.devices())\\n",
        "mesh = jax.make_mesh([(n, 1), ('fsdp', 'tp')], axis_types=(jax.sharding.AxisType.Auto,) * 2)\\n",
        "\\n",
        "model_path = snapshot_download(repo_id=base_model_id, ignore_patterns=['*.pth'])\\n",
        "with mesh:\\n",
        "    model = qwen_params.create_model_from_safe_tensors(model_path, config, mesh, dtype=jnp.bfloat16)\\n",
        "\\n",
        "# Apply LoRA (must match training config)\\n",
        "print('Applying LoRA...')\\n",
        "lora_provider = qwix_lora.LoraProvider(\\n",
        "    module_path='.*gate_proj|.*down_proj|.*up_proj',\\n",
        "    rank=16, alpha=32.0,\\n",
        ")\\n",
        "model_input = model.get_model_input()\\n",
        "lora_model = qwix_lora.apply_lora_to_model(model, lora_provider, rngs=nnx.Rngs(0), **model_input)\\n",
        "\\n",
        "# Load trained checkpoint\\n",
        "ckpt_dir = f'/content/checkpoints/{MODEL_VARIANT}'\\n",
        "print(f'Loading {MODEL_VARIANT} checkpoint from {ckpt_dir}...')\\n",
        "checkpointer = ocp.StandardCheckpointer()\\n",
        "if os.path.exists(ckpt_dir) and any(os.scandir(ckpt_dir)):\\n",
        "    restored = checkpointer.restore(ckpt_dir)\\n",
        "    nnx.update(lora_model, restored)\\n",
        "    print(f'Checkpoint {MODEL_VARIANT} loaded successfully!')\\n",
        "else:\\n",
        "    print(f'WARNING: {ckpt_dir} is empty — running with untrained LoRA weights!')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Inference — autoregressive generation with streaming\\n",
        "from tunix.sft import utils as tunix_utils\\n",
        "import sys\\n",
        "\\n",
        "def generate(question, max_new_tokens=512):\\n",
        "    prompt = f'<|im_start|>user\\\\n{question}<|im_end|>\\\\n<|im_start|>assistant\\\\n'\\n",
        "    input_ids = jnp.array([tokenizer.encode(prompt)])\\n",
        "    current_ids = input_ids\\n",
        "    print(f'Q: {question}')\\n",
        "    print(f'A: ', end='')\\n",
        "    \\n",
        "    for _ in range(max_new_tokens):\\n",
        "        seq_len = current_ids.shape[-1]\\n",
        "        mask = jnp.ones_like(current_ids, dtype=jnp.bool_)\\n",
        "        positions = tunix_utils.build_positions_from_mask(mask)\\n",
        "        attn_mask = tunix_utils.make_causal_attn_mask(mask)\\n",
        "        \\n",
        "        logits = lora_model(\\n",
        "            input_tokens=current_ids,\\n",
        "            input_mask=jnp.ones_like(current_ids),\\n",
        "            positions=positions,\\n",
        "            attention_mask=attn_mask,\\n",
        "        )\\n",
        "        next_token = jnp.argmax(logits[0, -1, :])\\n",
        "        \\n",
        "        if next_token == tokenizer.eos_token_id or int(next_token) == tokenizer.convert_tokens_to_ids('<|im_end|>'):\\n",
        "            break\\n",
        "        \\n",
        "        current_ids = jnp.concatenate([current_ids, jnp.array([[next_token]])], axis=1)\\n",
        "        char = tokenizer.decode([int(next_token)])\\n",
        "        sys.stdout.write(char)\\n",
        "        sys.stdout.flush()\\n",
        "    print('\\\\n')\\n",
        "\\n",
        "# Test prompts\\n",
        "questions = [\\n",
        "    'What is 25% of 200?',\\n",
        "    'Solve: 5x + 12 = 32.',\\n",
        "    'If a_n = 2 * a_{n-1} + 3 and a_1 = 1, find a_4.',\\n",
        "    'A store sells notebooks for $3 and pens for $1.50. If Maria buys 4 notebooks and 6 pens, how much does she spend?',\\n",
        "    'How does the S-CoT reasoning framework differ from flat chain-of-thought?',\\n",
        "]\\n",
        "\\n",
        "for q in questions:\\n",
        "    generate(q)\\n",
        "    print('=' * 60)"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}

with open("colab_inference.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook colab_inference.ipynb successfully generated!")
