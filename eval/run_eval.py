"""
Run GSM8K evaluation on any checkpoint.
Usage:
  python eval/run_eval.py \
      --model_path gs://YOUR_BUCKET/checkpoints/grpo-scot-1.5b/ \
      --benchmark  gsm8k \
      --output     eval/results_grpo.json
"""
import argparse, json, re
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return text.strip().split("\n")[-1]  # fallback: last line

def evaluate_gsm8k(model, tokenizer, device, n=1319) -> dict:
    ds   = load_dataset("gsm8k", "main", split="test")
    correct, total = 0, 0
    for row in tqdm(list(ds)[:n], desc="GSM8K"):
        gt       = row["answer"].split("####")[-1].strip()
        inputs   = tokenizer(row["question"], return_tensors="pt").to(device)
        outputs  = model.generate(**inputs, max_new_tokens=1024)
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred     = extract_answer(output_text)
        correct += int(pred.lower() == gt.lower())
        total   += 1
    return {"gsm8k_pass@1": round(correct / total, 4), "correct": correct, "total": total}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--benchmark", default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)

    if args.benchmark == "gsm8k":
        results = evaluate_gsm8k(model, tokenizer, device)
    
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
