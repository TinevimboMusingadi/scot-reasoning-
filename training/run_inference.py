"""
Run inference on the TPU using the trained LoRA model.
Generates answers for sample questions and saves results to JSON.

Usage (on TPU VM):
  source .venv/bin/activate
  python ~/scot/training/run_inference.py --model scot
  python ~/scot/training/run_inference.py --model flat
"""
import os, json, argparse, time
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import snapshot_download
from tunix.models.qwen2 import model as qwen_lib
from tunix.models.qwen2 import params as qwen_params
from tunix.sft import utils as tunix_utils
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

TEST_QUESTIONS = [
    "What is 25% of 200?",
    "Solve: 5x + 12 = 32.",
    "If a_n = 2 * a_{n-1} + 3 and a_1 = 1, find a_4.",
    "A store sells notebooks for $3 and pens for $1.50. If Maria buys 4 notebooks and 6 pens, how much does she spend?",
    "How does the S-CoT reasoning framework differ from flat chain-of-thought?",
    "There are 5 red balls and 3 blue balls in a bag. What is the probability of drawing a red ball?",
    "If a train travels at 60 mph for 2.5 hours, how far does it go?",
    "Explain what a derivative is in calculus.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["scot", "flat"], default="scot")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output-dir", default=os.path.expanduser("~/scot/outputs"))
    args = parser.parse_args()

    variant = f"sft-{args.model}"
    ckpt_dir = os.path.join(args.output_dir, variant)

    print(f"=== Inference: {variant} ===")
    print(f"JAX devices: {jax.devices()}")

    # 1. Load tokenizer
    from transformers import AutoTokenizer
    base_model_id = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # Add S-CoT tokens if running scot model
    if args.model == "scot":
        tokenizer.add_tokens(SCOT_TOKENS, special_tokens=True)

    # 2. Build base model
    print("Building base model...")
    config = qwen_lib.ModelConfig.qwen2p5_3b()
    n = len(jax.devices())
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((n, 1)),
        ('fsdp', 'tp')
    )

    model_path = snapshot_download(repo_id=base_model_id, ignore_patterns=["*.pth"])
    with mesh:
        model = qwen_params.create_model_from_safe_tensors(
            model_path, config, mesh, dtype=jnp.bfloat16
        )

    # 3. Apply LoRA
    print("Applying LoRA...")
    lora_provider = qwix.LoraProvider(
        module_path=".*gate_proj|.*down_proj|.*up_proj",
        rank=16, alpha=32.0,
    )
    model_input = model.get_model_input()
    lora_model = qwix.apply_lora_to_model(
        model, lora_provider, rngs=nnx.Rngs(0), **model_input
    )

    # 4. Load checkpoint
    print(f"Loading checkpoint from {ckpt_dir}...")
    from orbax import checkpoint as ocp
    checkpointer = ocp.StandardCheckpointer()
    if os.path.exists(ckpt_dir) and any(os.scandir(ckpt_dir)):
        restored = checkpointer.restore(ckpt_dir)
        nnx.update(lora_model, restored)
        print("Checkpoint loaded!")
    else:
        print(f"WARNING: {ckpt_dir} empty — using untrained LoRA weights")

    # 5. Generate answers
    results = []
    for q in TEST_QUESTIONS:
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = jnp.array([tokenizer.encode(prompt)])
        current_ids = input_ids

        print(f"\nQ: {q}")
        print("A: ", end="", flush=True)

        generated_tokens = []
        start_time = time.time()

        for step in range(args.max_tokens):
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
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if next_token in [eos_id, im_end_id]:
                break

            generated_tokens.append(next_token)
            current_ids = jnp.concatenate(
                [current_ids, jnp.array([[next_token]])], axis=1
            )
            char = tokenizer.decode([next_token])
            print(char, end="", flush=True)

        elapsed = time.time() - start_time
        answer = tokenizer.decode(generated_tokens)
        print(f"\n  [{len(generated_tokens)} tokens in {elapsed:.1f}s]")

        results.append({
            "question": q,
            "answer": answer,
            "model": variant,
            "num_tokens": len(generated_tokens),
            "time_seconds": round(elapsed, 2),
        })

    # 6. Save results
    results_path = os.path.join(args.output_dir, f"inference_{args.model}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
