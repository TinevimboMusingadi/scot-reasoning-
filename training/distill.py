"""
Logit distillation: transfer reasoning from Qwen2.5-7B teacher → 1.5B student.
Both models must already be SFT-trained on S-CoT traces.
Run ON the TPU VM.

Usage:
  python ~/scot/training/distill.py \
      --teacher_ckpt gs://YOUR_BUCKET/checkpoints/sft-scot-7b/ \
      --student_ckpt gs://YOUR_BUCKET/checkpoints/sft-scot-1.5b/ \
      --data         gs://YOUR_BUCKET/scot_traces.jsonl \
      --output       gs://YOUR_BUCKET/checkpoints/distilled-1.5b/
"""
import argparse
import jax, jax.numpy as jnp, optax
from tunix.distillation import distillation_trainer, distillation_config
from tunix.distillation.strategies import logit_strategy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--student_ckpt", required=True)
    parser.add_argument("--data",         required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--steps",        type=int,   default=400)
    parser.add_argument("--temperature",  type=float, default=3.0)
    parser.add_argument("--alpha",        type=float, default=0.7)
    args = parser.parse_args()

    n = len(jax.devices())
    MESH = [(1, n), ("fsdp", "tp")]
    mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * 2)

    # --- Load teacher (7B) and student (1.5B) ---
    # TODO: Load models using tunix logic (e.g. from AutoModel or qwen_params)
    teacher = None  
    student = None  
    train_data = [] 

    config = distillation_config.DistillationConfig(
        strategy=logit_strategy.LogitStrategy(
            temperature=args.temperature,
            alpha=args.alpha,           # weight on KL(teacher||student) loss
        ),
        optimizer=optax.adamw(learning_rate=1e-4),
        num_steps=args.steps,
    )

    trainer = distillation_trainer.DistillationTrainer(
        teacher_model=teacher,   
        student_model=student,   
        config=config,
        mesh=mesh,
    )

    # NOTE: pass only the student training data — teacher generates soft labels on-the-fly
    trainer.train(train_ds=train_data)
    trainer.save(args.output)
    print(f"Distilled checkpoint saved to {args.output}")

if __name__ == "__main__":
    main()
