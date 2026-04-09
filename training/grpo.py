"""
GRPO fine-tuning with mode-aware rewards.
Start from the distilled checkpoint.
Run ON the TPU VM.

Usage:
  python ~/scot/training/grpo.py \
      --base_ckpt gs://YOUR_BUCKET/checkpoints/distilled-1.5b/ \
      --data      gs://YOUR_BUCKET/scot_traces.jsonl \
      --output    gs://YOUR_BUCKET/checkpoints/grpo-scot-1.5b/
"""
import argparse, os
import jax
from tunix.rl import grpo_learner
from tunix import GRPOConfig
from rewards import total_reward

def reward_fn(completions: list[str], ground_truths: list[str]) -> list[float]:
    return [total_reward(c, gt) for c, gt in zip(completions, ground_truths)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--data",      required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--steps",     type=int,   default=300)
    parser.add_argument("--group_size",type=int,   default=8)
    args = parser.parse_args()

    n = len(jax.devices())
    MESH = [(1, n), ("fsdp", "tp")]
    mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * 2)

    # Load base model from checkpoint 
    # TODO: load model from args.base_ckpt
    model = None

    config = GRPOConfig(
        learning_rate=5e-6,
        num_iterations=args.steps,
        num_generations=args.group_size,   # samples per prompt for GRPO baseline
        beta=0.04, # equivalent to kl_coeff
    )

    learner = grpo_learner.GRPOLearner(
        rl_cluster=None, # TODO: Setup rl_cluster with model roles
        algo_config=config,
        reward_fns=[reward_fn],
    )

    train_data = [] # TODO: load data appropriately
    learner.train(train_ds=train_data)
    # Note: saving is typically handled via checkpoint manager in RLTrainingConfig 
    print(f"GRPO training completed.")

if __name__ == "__main__":
    main()
