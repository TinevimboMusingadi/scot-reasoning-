"""
GRPO fine-tuning with mode-aware rewards using Tunix.
Start from the SFT-trained checkpoint.
Run ON the TPU VM.
"""
import argparse, os, json
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tunix.rl import grpo_learner, rl_cluster
from tunix import GRPOConfig, ClusterConfig, RLTrainingConfig, RolloutConfig
from tunix.models.qwen2 import model as qwen_lib
from tunix.models.qwen2 import params_safetensors as qwen_params
from transformers import AutoTokenizer
from rewards import total_reward

def reward_fn(completions: list[str], ground_truths: list[str]) -> list[float]:
    """Calculate reward for each completion."""
    return [total_reward(c, gt) for c, gt in zip(completions, ground_truths)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", required=True, help="Path to SFT checkpoint")
    parser.add_argument("--data",      required=True, help="Path to scot_traces.jsonl")
    parser.add_argument("--output",    required=True, help="Output directory for RL weights")
    parser.add_argument("--steps",     type=int,   default=300)
    parser.add_argument("--group_size",type=int,   default=8)
    parser.add_argument("--model_id",  default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    n = len(jax.devices())
    MESH = [(1, n), ("fsdp", "tp")]
    mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * 2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # Special tokens already added during SFT, but we ensure they are here
    SCOT_TOKENS = ["<reasoning>", "</reasoning>", "<meta_reasoning>", "</meta_reasoning>", 
                   "<abduction>", "</abduction>", "<decompose>", "</decompose>", 
                   "<deduction>", "</deduction>", "<induction>", "</induction>", 
                   "<analogy>", "</analogy>", "<causal>", "</causal>", "<answer>", "</answer>"]
    tokenizer.add_special_tokens({"additional_special_tokens": SCOT_TOKENS})

    # 1. Setup Cluster Config
    c_config = ClusterConfig(
        role_to_mesh={rl_cluster.Role.ACTOR: mesh, rl_cluster.Role.REFERENCE: mesh},
        training_config=RLTrainingConfig(
            eval_every_n_steps=50,
            max_steps=args.steps,
            actor_optimizer=optax.adamw(learning_rate=5e-6),
        ),
        rollout_config=RolloutConfig(
            max_tokens_to_generate=512,
            temperature=0.9,
            top_p=0.95,
        )
    )

    # 2. Load Model (Actor and Reference)
    config = qwen_lib.ModelConfig.qwen2_5_1_5b()
    config.vocab_size = len(tokenizer)
    
    # In Tunix, RLCluster can take a factory function or a model instance
    def model_factory(mesh):
        with mesh:
            # Note: We'd typically load the SFT weights here using orbax
            m = qwen_params.create_model_from_safe_tensors(
                args.base_ckpt, config, mesh, dtype=jnp.bfloat16
            )
            return m

    cluster = rl_cluster.RLCluster(
        actor=model_factory(mesh),
        reference=model_factory(mesh),
        tokenizer=tokenizer,
        cluster_config=c_config,
    )

    # 3. Setup Learner
    algo_config = GRPOConfig(
        num_generations=args.group_size,
        num_iterations=1,
        beta=0.04,
    )

    learner = grpo_learner.GRPOLearner(
        rl_cluster=cluster,
        algo_config=algo_config,
        reward_fns=[reward_fn],
    )

    # 4. Load Data
    import jsonlines
    train_data = []
    with jsonlines.open(args.data) as reader:
        for row in reader:
            # GRPO needs a 'prompt' field
            train_data.append({"prompt": f"<|im_start|>user\n{row['problem']}<|im_end|>\n<|im_start|>assistant\n", 
                               "ground_truth": row["answer"]})

    print(f"Starting GRPO training with {len(train_data)} prompts.")
    learner.train(train_ds=train_data)
    
    # 5. Save final checkpoint
    save_path = args.output
    os.makedirs(save_path, exist_ok=True)
    from orbax import checkpoint as ocp
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(save_path, nnx.state(cluster.actor))
    print(f"GRPO training completed. Saved to {save_path}")

if __name__ == "__main__":
    main()
