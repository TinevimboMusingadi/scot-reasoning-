"""
SFT fine-tuning of Qwen2.5-3B on flat <think> traces using Tunix.
Run ON the TPU VM after setup_tpu.sh completes.

Usage (on TPU VM):
  source .venv/bin/activate
  python ~/scot/training/sft_flat.py
"""
import os, json, argparse
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from huggingface_hub import snapshot_download
from tunix.models.qwen2 import model as qwen_lib
from tunix.models.qwen2 import params as qwen_params
from tunix.sft import peft_trainer
from tunix.sft import utils as tunix_utils
import qwix

def build_prompt(problem: str, trace: str) -> str:
    return f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n{trace}<|im_end|>"

def load_dataset_from_jsonl(path: str, tokenizer, max_seq_len: int):
    """Load JSONL, tokenise, return list of dicts with input_tokens + input_mask."""
    import jsonlines, numpy as np
    examples = []
    with jsonlines.open(path) as reader:
        for row in reader:
            text = build_prompt(row["problem"], row["flat_trace"])
            tokens = tokenizer.encode(text)[:max_seq_len]
            pad_len = max_seq_len - len(tokens)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            input_tokens = tokens + [pad_id] * pad_len
            input_mask   = [1] * len(tokens) + [0] * pad_len
            examples.append({
                "input_tokens": np.array(input_tokens, dtype=np.int32),
                "input_mask":   np.array(input_mask,   dtype=np.int32),
            })
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default=os.path.expanduser("~/scot/data/full_run/flat_traces.jsonl"))
    parser.add_argument("--output", default=os.path.expanduser("~/scot/outputs/sft-flat"))
    parser.add_argument("--model",  default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--steps",  type=int, default=500)
    parser.add_argument("--lr",     type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    # Mesh — v5p-8 has 4 or 8 chips. Use FSDP instead of TP to avoid head divisibility errors.
    n = len(jax.devices())
    MESH = [(n, 1), ("fsdp", "tp")]
    mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * 2)

    # Download model weights
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model_path = snapshot_download(repo_id=args.model, ignore_patterns=["*.pth"])

    config = qwen_lib.ModelConfig.qwen2p5_3b()
    with mesh:
        model = qwen_params.create_model_from_safe_tensors(
            model_path, config, mesh, dtype=jnp.bfloat16
        )

    # Apply LoRA — use Tunix's native get_model_input() as shown in the official docs.
    lora_provider = qwix.LoraProvider(
        module_path=".*gate_proj|.*down_proj|.*up_proj",
        rank=args.lora_rank,
        alpha=args.lora_rank * 2,
    )
    model_input = model.get_model_input()
    lora_model = qwix.apply_lora_to_model(
        model, lora_provider, **model_input
    )

    # Shard the LoRA state properly across the mesh (from official Tunix LoRA docs)
    with mesh:
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(lora_model, sharded_state)

    save_path = args.output
    # Safe Restoration logic!
    from orbax import checkpoint as ocp
    checkpointer = ocp.StandardCheckpointer()
    if os.path.isdir(save_path) and any(os.scandir(save_path)):
        try:
            print(f"[*] Resuming weights proactively from resilient backup in {save_path}...")
            restored = checkpointer.restore(save_path)
            nnx.update(lora_model, restored)
        except Exception as e:
            print(f"[!] Warning: Safe Resumption failed natively. Proceeding from baseline... Error: {e}")

    # Dataset
    MAX_SEQ_LEN = 2048
    train_data  = load_dataset_from_jsonl(args.data, tokenizer, MAX_SEQ_LEN)
    print(f"Loaded {len(train_data)} training examples.")

    # Determine the pad token ID for building masks in input_fn
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # gen_model_input_fn — follows the official Tunix pattern from the docs.
    # Uses tunix.sft.utils to build correct positions and causal attention masks.
    def input_fn(x):
        mask = x["input_tokens"] != pad_id
        return {
            "input_tokens": x["input_tokens"],
            "input_mask": x["input_mask"],
            "positions": tunix_utils.build_positions_from_mask(mask),
            "attention_mask": tunix_utils.make_causal_attn_mask(mask),
        }

    # Trainer Config
    t_config  = peft_trainer.TrainingConfig(
        eval_every_n_steps=100, 
        max_steps=args.steps
    )

    trainer = peft_trainer.PeftTrainer(
        model=lora_model,
        optimizer=optax.adamw(learning_rate=args.lr),
        training_config=t_config,
    )

    trainer.with_gen_model_input_fn(input_fn)
    trainer.train(train_ds=train_data)

    # Save checkpoint
    from orbax import checkpoint as ocp
    checkpointer = ocp.StandardCheckpointer()
    os.makedirs(args.output, exist_ok=True)
    checkpointer.save(args.output, nnx.state(lora_model))
    print(f"Checkpoint saved to {args.output}")

if __name__ == "__main__":
    main()
