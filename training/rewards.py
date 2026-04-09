"""
Reward functions for GRPO training.
Imported by grpo.py — keep this file self-contained, no Tunix imports here.
"""
import re

MODE_TAGS = ["abduction", "decompose", "deduction", "induction", "analogy", "causal"]

def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else ""

def reward_correct(completion: str, ground_truth: str) -> float:
    return 1.0 if extract_answer(completion).lower() == ground_truth.lower() else 0.0

def reward_format(completion: str) -> float:
    has_reasoning = "<reasoning>" in completion and "</reasoning>" in completion
    has_answer    = "<answer>" in completion and "</answer>" in completion
    has_mode      = any(f"<{t}>" in completion for t in MODE_TAGS)
    has_meta      = "<meta_reasoning>" in completion
    return 1.0 if all([has_reasoning, has_answer, has_mode, has_meta]) else 0.0

def reward_mode_diversity(completion: str) -> float:
    count = sum(1 for t in MODE_TAGS if f"<{t}>" in completion)
    return min(count / 3.0, 1.0)

def reward_meta_quality(completion: str) -> float:
    meta_blocks = re.findall(
        r"<meta_reasoning>(.*?)</meta_reasoning>", completion, re.DOTALL
    )
    if not meta_blocks:
        return 0.0
    opens_ok      = completion.find("<meta_reasoning>") < 200
    all_substance = all(len(b.split()) >= 15 for b in meta_blocks)
    mode_count    = sum(1 for t in MODE_TAGS if f"<{t}>" in completion)
    count_ok      = len(meta_blocks) >= mode_count + 1
    return round((int(opens_ok) + int(all_substance) + int(count_ok)) / 3.0, 4)

def total_reward(completion: str, ground_truth: str) -> float:
    return round(
        0.45 * reward_correct(completion, ground_truth) +
        0.15 * reward_format(completion) +
        0.15 * reward_mode_diversity(completion) +
        0.25 * reward_meta_quality(completion),
        4
    )
