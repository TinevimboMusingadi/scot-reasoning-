"""
Generates colab_inference.ipynb for S-CoT / Flat model inference.

The notebook JSON spec requires each line in cell "source" to end with
a literal newline character (\n). Python's json.dump handles this correctly
when we use regular \n in our Python strings.
"""
import json

def md_cell(lines):
    """Create a markdown cell from a list of lines."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" if not line.endswith("\n") else line for line in lines[:-1]] + [lines[-1]]
    }

def code_cell(code_text):
    """Create a code cell from a multi-line string."""
    lines = code_text.split("\n")
    # Add \n to all lines except the last
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

notebook = {
    "cells": [
        md_cell([
            "# S-CoT Distilled Model \u2014 Inference Notebook",
            "",
            "This notebook downloads your TPU-trained LoRA checkpoints from GCS, injects them into the **Qwen2.5-3B-Instruct** base model, and runs inference.",
            "",
            "**Runtime:** Select **T4 or L4 GPU** in Colab.",
            "",
            "**Two models available:**",
            "- `sft-scot`: S-CoT structured reasoning (loss=1.27, ppl=3.55)",
            "- `sft-flat`: Flat baseline (loss=0.275, ppl=1.32)",
        ]),

        code_cell("""\
# Cell 1: Install dependencies
!pip install -q 'numpy<2.0.0'
!pip install -q 'google-tunix[prod]'
!pip install -q -U 'transformers>=4.45,<=4.57.1'
!pip install -q wandb huggingface_hub gcsfs datasets evaluate tqdm peft

# Restart runtime so numpy C-extensions reload cleanly
import os
os.kill(os.getpid(), 9)"""),

        code_cell("""\
# Cell 2: Authenticate and download checkpoints from GCS
from google.colab import auth
auth.authenticate_user()

!mkdir -p /content/checkpoints/sft-scot
!mkdir -p /content/checkpoints/sft-flat
!gsutil -m cp -r gs://tpu-builder1-scot-checkpoints/sft-scot/* /content/checkpoints/sft-scot/ 2>/dev/null || echo 'S-CoT checkpoint not found in bucket'
!gsutil -m cp -r gs://tpu-builder1-scot-checkpoints/sft-flat/* /content/checkpoints/sft-flat/ 2>/dev/null || echo 'Flat checkpoint not found in bucket'

import os
for variant in ['sft-scot', 'sft-flat']:
    path = f'/content/checkpoints/{variant}'
    files = os.listdir(path) if os.path.exists(path) else []
    print(f'=== {variant} === ({len(files)} files)')
    for f in files[:10]:
        print(f'  {f}')"""),

        code_cell("""\
#@title Cell 3: Select Model to Load { run: "auto" }
MODEL_VARIANT = 'sft-scot' #@param ['sft-scot', 'sft-flat']

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from tunix.models.qwen2 import model as qwen_model
from tunix.models.qwen2 import params as qwen_params
from qwix._src.providers import lora as qwix_lora
from flax import nnx
from orbax import checkpoint as ocp
from huggingface_hub import snapshot_download
import os

base_model_id = 'Qwen/Qwen2.5-3B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print('Building base model...')
config = qwen_model.ModelConfig.qwen2p5_3b()
devices = jax.devices()
mesh = jax.sharding.Mesh(
    jnp.array(devices).reshape((len(devices), 1)),
    ('fsdp', 'tp')
)

model_path = snapshot_download(repo_id=base_model_id, ignore_patterns=['*.pth'])
with mesh:
    model = qwen_params.create_model_from_safe_tensors(
        model_path, config, mesh, dtype=jnp.bfloat16
    )

print('Applying LoRA...')
lora_provider = qwix_lora.LoraProvider(
    module_path='.*gate_proj|.*down_proj|.*up_proj',
    rank=16,
    alpha=32.0,
)
model_input = model.get_model_input()
lora_model = qwix_lora.apply_lora_to_model(
    model, lora_provider, rngs=nnx.Rngs(0), **model_input
)

ckpt_dir = f'/content/checkpoints/{MODEL_VARIANT}'
print(f'Loading {MODEL_VARIANT} checkpoint from {ckpt_dir}...')
checkpointer = ocp.StandardCheckpointer()
if os.path.exists(ckpt_dir) and any(os.scandir(ckpt_dir)):
    restored = checkpointer.restore(ckpt_dir)
    nnx.update(lora_model, restored)
    print(f'Checkpoint {MODEL_VARIANT} loaded successfully!')
else:
    print(f'WARNING: {ckpt_dir} is empty. Running with untrained LoRA weights!')"""),

        code_cell("""\
# Cell 4: Inference with streaming
from tunix.sft import utils as tunix_utils
import sys

def generate(question, max_new_tokens=512):
    prompt = f'<|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n'
    input_ids = jnp.array([tokenizer.encode(prompt)])
    current_ids = input_ids

    print(f'Q: {question}')
    print('A: ', end='')

    for _ in range(max_new_tokens):
        mask = jnp.ones_like(current_ids, dtype=jnp.bool_)
        positions = tunix_utils.build_positions_from_mask(mask)
        attn_mask = tunix_utils.make_causal_attn_mask(mask)

        logits = lora_model(
            input_tokens=current_ids,
            input_mask=jnp.ones_like(current_ids),
            positions=positions,
            attention_mask=attn_mask,
        )
        next_token = jnp.argmax(logits[0, -1, :])

        if int(next_token) in [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids('<|im_end|>'),
        ]:
            break

        current_ids = jnp.concatenate(
            [current_ids, jnp.array([[next_token]])], axis=1
        )
        char = tokenizer.decode([int(next_token)])
        sys.stdout.write(char)
        sys.stdout.flush()

    print('\\n')

questions = [
    'What is 25% of 200?',
    'Solve: 5x + 12 = 32.',
    'If a_n = 2 * a_{n-1} + 3 and a_1 = 1, find a_4.',
    'A store sells notebooks for $3 and pens for $1.50. If Maria buys 4 notebooks and 6 pens, how much does she spend?',
    'How does the S-CoT reasoning framework differ from flat chain-of-thought?',
]

for q in questions:
    generate(q)
    print('=' * 60)"""),
    ],
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open("colab_inference.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook colab_inference.ipynb successfully generated!")
