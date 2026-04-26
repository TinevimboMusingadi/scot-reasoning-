"""
Generates colab_inference.ipynb — Runs inference on Colab TPU runtime.

Tunix is TPU-only, so the notebook MUST use a Colab TPU runtime.
Runtime > Change runtime type > TPU v2
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
            "# \U0001f9e0 S-CoT Distilled Model \u2014 TPU Inference",
            "",
            "Run inference on your distilled Qwen2.5-3B LoRA models directly on a **Colab TPU**.",
            "",
            "> \u26a0\ufe0f **IMPORTANT:** Go to **Runtime \u2192 Change runtime type \u2192 TPU v2**",
            "> before running this notebook. Tunix is a TPU-only framework.",
            "",
            "**Two models available:**",
            "| Model | Final Loss | Perplexity | Dataset |",
            "|-------|-----------|------------|---------|",
            "| `sft-scot` (S-CoT reasoning) | 1.27 | 3.55 | 3,817 |",
            "| `sft-flat` (Flat baseline) | 0.275 | 1.32 | 3,681 |",
        ]),

        code_cell("""\
# Cell 1: Install dependencies with uv (faster + better resolution)
!pip install -q uv
!uv pip install --system 'google-tunix[prod]' 'transformers>=4.45,<=4.57.1'

# Restart runtime so C-extensions reload cleanly
import os
os.kill(os.getpid(), 9)"""),

        code_cell("""\
# Cell 2: Verify TPU & authenticate
import jax
devices = jax.devices()
print(f"JAX devices: {devices}")
print(f"Device type: {devices[0].platform}")

assert devices[0].platform == 'tpu', (
    "\\n\\n\u274c You are NOT on a TPU runtime!\\n"
    "Go to Runtime > Change runtime type > TPU v2\\n"
    "Then restart and re-run from Cell 1."
)
print(f"\\n\u2705 TPU detected! {len(devices)} cores available.")

# Authenticate for GCS access
from google.colab import auth
auth.authenticate_user()
print("\u2705 Authenticated with Google Cloud.")

# Nuke any expired HuggingFace tokens from disk
import os, shutil
for p in [
    os.path.expanduser('~/.cache/huggingface/token'),
    os.path.expanduser('~/.huggingface/token'),
]:
    if os.path.exists(p):
        os.remove(p) if os.path.isfile(p) else shutil.rmtree(p)
        print(f'Removed expired HF token: {p}')

os.environ.pop('HF_TOKEN', None)
os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)
print("\u2705 HuggingFace auth cleared.")\
"""),

        code_cell("""\
# Cell 3: Download checkpoints from GCS
import os

!mkdir -p /content/checkpoints/sft-scot
!mkdir -p /content/checkpoints/sft-flat

print("Downloading from GCS bucket...")
# Orbax may save as .orbax-checkpoint-tmp if async finalization fails,
# so we download everything and check both paths
!gsutil -m cp -r 'gs://tpu-builder1-scot-checkpoints/sft-scot/**' /content/checkpoints/sft-scot/ 2>/dev/null || true
!gsutil -m cp -r 'gs://tpu-builder1-scot-checkpoints/sft-scot.orbax-checkpoint-tmp/**' /content/checkpoints/sft-scot/ 2>/dev/null || true
!gsutil -m cp -r 'gs://tpu-builder1-scot-checkpoints/sft-flat/**' /content/checkpoints/sft-flat/ 2>/dev/null || true
!gsutil -m cp -r 'gs://tpu-builder1-scot-checkpoints/sft-flat.orbax-checkpoint-tmp/**' /content/checkpoints/sft-flat/ 2>/dev/null || true

# Show what we got
print("\\n" + "="*50)
for variant in ['sft-scot', 'sft-flat']:
    path = f'/content/checkpoints/{variant}'
    if os.path.exists(path):
        count = 0
        total = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                count += 1
                total += os.path.getsize(os.path.join(root, f))
        print(f"  {variant}: {count} files, {total:,} bytes")
    else:
        print(f"  {variant}: not found")\
"""),

        code_cell("""\
#@title Cell 4: Load Model & Checkpoint { run: "auto" }
MODEL_VARIANT = 'sft-scot' #@param ['sft-scot', 'sft-flat']

import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from tunix.models.qwen2 import model as qwen_lib
from tunix.models.qwen2 import params as qwen_params
from flax import nnx
from orbax import checkpoint as ocp
from huggingface_hub import snapshot_download
import qwix
import os

# S-CoT special tokens
SCOT_TOKENS = [
    "<reasoning>", "</reasoning>",
    "<meta_reasoning>", "</meta_reasoning>",
    "<abduction>", "</abduction>",
    "<decompose>", "</decompose>",
    "<deduction>", "</deduction>",
    "<induction>", "</induction>",
    "<analogy>", "</analogy>",
    "<causal>", "</causal>",
    "<answer>", "</answer>",
]

# 1. Load tokenizer
base_model_id = 'Qwen/Qwen2.5-3B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=False)

if MODEL_VARIANT == 'sft-scot':
    tokenizer.add_tokens(SCOT_TOKENS, special_tokens=True)
    print(f"Added {len(SCOT_TOKENS)} S-CoT tokens.")

# 2. Build base model on TPU — use np.array, NOT jnp.array for devices!
print("\\nBuilding Qwen2.5-3B on TPU...")
config = qwen_lib.ModelConfig.qwen2p5_3b()
devices = jax.devices()
n = len(devices)
mesh = jax.sharding.Mesh(
    np.array(devices).reshape((n, 1)),
    ('fsdp', 'tp')
)

model_path = snapshot_download(repo_id=base_model_id, ignore_patterns=['*.pth'], token=False)
with mesh:
    model = qwen_params.create_model_from_safe_tensors(
        model_path, config, mesh, dtype=jnp.bfloat16
    )
print("\u2705 Base model loaded.")

# 3. Apply LoRA
print("Applying LoRA (rank=16, alpha=32)...")
lora_provider = qwix.LoraProvider(
    module_path='.*gate_proj|.*down_proj|.*up_proj',
    rank=16,
    alpha=32.0,
)
model_input = model.get_model_input()
lora_model = qwix.apply_lora_to_model(
    model, lora_provider, rngs=nnx.Rngs(0), **model_input
)

# 4. Load checkpoint
ckpt_dir = f'/content/checkpoints/{MODEL_VARIANT}'
print(f"\\nLoading checkpoint from {ckpt_dir}...")
checkpointer = ocp.StandardCheckpointer()
if os.path.exists(ckpt_dir) and any(os.scandir(ckpt_dir)):
    restored = checkpointer.restore(ckpt_dir)
    nnx.update(lora_model, restored)
    print(f"\u2705 {MODEL_VARIANT} checkpoint loaded!")
else:
    print(f"\u26a0\ufe0f {ckpt_dir} is empty — running with UNTRAINED LoRA weights!")

print(f"\\n\U0001f680 Ready for inference ({MODEL_VARIANT}).")\
"""),

        code_cell("""\
# Cell 5: Run inference
from tunix.sft import utils as tunix_utils
import time, sys

def generate(question, max_new_tokens=512):
    prompt = f'<|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n'
    input_ids = jnp.array([tokenizer.encode(prompt)])
    current_ids = input_ids

    print(f'\\n{\"=\"*60}')
    print(f'\U0001f4ac Q: {question}')
    print(f'\U0001f9e0 A: ', end='')

    generated_tokens = []
    start = time.time()

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
        next_token = int(jnp.argmax(logits[0, -1, :]))

        eos_id = tokenizer.eos_token_id
        im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
        if next_token in [eos_id, im_end_id]:
            break

        generated_tokens.append(next_token)
        current_ids = jnp.concatenate(
            [current_ids, jnp.array([[next_token]])], axis=1
        )
        char = tokenizer.decode([next_token])
        sys.stdout.write(char)
        sys.stdout.flush()

    elapsed = time.time() - start
    print(f'\\n\\n\U0001f4ca [{len(generated_tokens)} tokens in {elapsed:.1f}s]')
    return tokenizer.decode(generated_tokens)

# --- Test questions ---
questions = [
    'What is 25% of 200?',
    'Solve: 5x + 12 = 32.',
    'If a_n = 2 * a_{n-1} + 3 and a_1 = 1, find a_4.',
    'A store sells notebooks for $3 and pens for $1.50. If Maria buys 4 notebooks and 6 pens, how much does she spend?',
    'How does the S-CoT reasoning framework differ from flat chain-of-thought?',
]

results = []
for q in questions:
    answer = generate(q)
    results.append({'question': q, 'answer': answer})\
"""),

        code_cell("""\
# Cell 6: Pretty-print results
from IPython.display import HTML, display

display(HTML(f'<h2 style="color:#e94560;">\U0001f9e0 {MODEL_VARIANT} Results</h2>'))
for r in results:
    display(HTML(f'''
    <div style="border:1px solid #555; border-radius:10px; padding:18px; margin:14px 0;
                background:linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
        <div style="color:#e94560; font-weight:bold; font-size:15px;
                    border-bottom:1px solid #333; padding-bottom:8px; margin-bottom:10px;">
            \U0001f4ac Q: {r["question"]}
        </div>
        <div style="color:#eee; white-space:pre-wrap;
                    font-family:monospace; font-size:13px; line-height:1.6;">
{r["answer"]}
        </div>
    </div>
    '''))

print("\\n\u2705 Done! Switch MODEL_VARIANT in Cell 4 to try the other model.")\
"""),
    ],
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
        "accelerator": "TPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open("colab_inference.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook generated!")
