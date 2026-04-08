# S-CoT Reasoning on Google Cloud TPU

This repository implements **Structured Chain-of-Thought (S-CoT)** distillation from a large reasoning model (via Gemini API) into a smaller Qwen student model (`Qwen2.5-1.5B-Instruct`), running on Google Cloud TPU v5p using the Tunix framework.

## Overview
1. **Trace Generation**: Uses `google.generativeai` to synthesize structured reasoning paths consisting of categorized cognitive modes (e.g., `<decompose>`, `<deduction>`) and routing components (`<meta_reasoning>`).
2. **Setup**: Connect to Google Cloud TPU (`my-tpu-node` in `us-east5-a`) and deploy Tunix.
3. **Training**: Performs Supervised Fine-Tuning (SFT), Logit Distillation, and Group Relative Policy Optimization (GRPO) directly on TPU.
4. **Evaluation**: Evaluates the model on reasoning benchmarks (e.g., GSM8K).

## Quick Start (Local Data Generation)
1. Install requirements: `pip install -r requirements.txt`
2. Populate `.env` with your API keys.
3. Run data generation: `python data/generate_traces.py`

## Dataset Statistics
We scale structured reasoning context significantly through teacher-model distillation.

| Dataset | Total Samples | Total Tokens | Avg / Sample |
| :--- | :--- | :--- | :--- |
| **Original GSM8K** | 7,473 | 448,443 | 60 |
| **S-CoT Traces** | 1,929 | 1,057,318 | 548 |
| **Flat Traces** | 1,876 | 527,888 | 281 |

*Audited using Gemini 1.5 Pro.*
