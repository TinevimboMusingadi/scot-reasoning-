"""
SFT fine-tuning of Qwen2.5-1.5B on structured S-CoT traces using Tunix.
Run ON the TPU VM after setup_tpu.sh completes.

Usage (on TPU VM):
  source .venv/bin/activate
  python ~/scot/training/sft_scot.py
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
import qwix

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

def build_prompt(problem: str, trace: str) -> str:
    return f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n{trace}<|im_end|>"

def load_dataset_from_jsonl(path: str, tokenizer, max_seq_len: int):
    """Load JSONL, tokenise, return list of dicts with input_tokens + input_mask."""
    import jsonlines, numpy as np
    examples = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
        
    with jsonlines.open(path) as reader:
        for row in reader:
            text = build_prompt(row["problem"], row["scot_trace"])
            tokens = tokenizer.encode(text)[:max_seq_len]
            pad_len = max_seq_len - len(tokens)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            input_tokens = tokens + [pad_id] * pad_len
            input_mask   = [1] * len(tokens) + [0] * pad_len
            examples.append({
                "input_tokens": np.array(input_tokens, dtype=np.int32),
                "input_mask":   np.array(input_mask,   dtype=np.int32),
                "positions":    np.arange(max_seq_len, dtype=np.int32),
            })
    return examples

def main():
    parser = argparse.ArgumentParser()
    # Updated to point to our local scot_traces.jsonl by default
    parser.add_argument("--data",   default=os.path.expanduser("~/scot/data/full_run/scot_traces.jsonl"))
    parser.add_argument("--output", default=os.path.expanduser("~/scot/outputs/sft-scot"))
    parser.add_argument("--model",  default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--steps",  type=int, default=500)
    parser.add_argument("--lr",     type=float, default=2e-5) # Slightly lower LR for reasoning distillation
    parser.add_argument("--lora_rank", type=int, default=16)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    # Mesh — v5p-8 has 4 or 8 chips. Use FSDP instead of TP to avoid head divisibility errors.
    n = len(jax.devices())
    MESH = [(n, 1), ("fsdp", "tp")]
    mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * 2)

    # Tokenizer setup
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Add special tokens for cognitive modes
    tokenizer.add_special_tokens({"additional_special_tokens": SCOT_TOKENS})
    
    # Download model weights
    model_path = snapshot_download(repo_id=args.model, ignore_patterns=["*.pth"])

    # Qwen Config
    config = qwen_lib.ModelConfig.qwen2p5_3b()
    # IMPORTANT: Update vocab size in config to match our expanded tokenizer
    config.vocab_size = len(tokenizer)
    
    with mesh:
        # Tunix handles initializing new weights (like embeddings) if size mismatch
        model = qwen_params.create_model_from_safe_tensors(
            model_path, config, mesh, dtype=jnp.bfloat16
        )

    # Apply QLoRA
    lora_provider = qwix.LoraProvider(
        module_path=".*gate_proj|.*down_proj|.*up_proj",
        rank=args.lora_rank,
        alpha=args.lora_rank * 2,
    )
    model_input = model.get_model_input(batch_size=1, seq_len=256)
    lora_model  = qwix.apply_lora_to_model(model, lora_provider, rngs=nnx.Rngs(0), **model_input)

    save_path = args.output
    # Safe Restoration logic!
    from orbax import checkpoint as ocp
    checkpointer = ocp.StandardCheckpointer()
    if os.path.isdir(save_path) and any(os.scandir(save_path)):
        try:
            print(f"[*] Resuming weights proactively from resilient backup in {save_path}...")
            restored = checkpointer.restore(save_path)
            nnx.update(lora_model, restored) # Safe injects structure directly
        except Exception as e:
            print(f"[!] Warning: Safe Resumption failed natively. Proceeding from baseline... Error: {e}")

    # Dataset
    MAX_SEQ_LEN = 2048
    train_data  = load_dataset_from_jsonl(args.data, tokenizer, MAX_SEQ_LEN)
    print(f"Loaded {len(train_data)} training examples.")

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

    trainer.with_gen_model_input_fn(lambda x: {"input_tokens": x["input_tokens"], "input_mask": x["input_mask"], "attention_mask": x["input_mask"], "positions": x["positions"]})
    trainer.train(train_ds=train_data)

    # Save checkpoint
    save_path = args.output
    os.makedirs(save_path, exist_ok=True)
    
    from orbax import checkpoint as ocp
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(save_path, nnx.state(lora_model))
    
    # Save the special combined tokenizer
    tokenizer.save_pretrained(save_path)
    print(f"SFT Training Complete. Checkpoint & Tokenizer saved to {save_path}")

if __name__ == "__main__":
    main()
