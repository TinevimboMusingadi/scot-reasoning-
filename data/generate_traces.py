"""
Generate structured S-CoT traces using Gemini.
Runs locally. Writes two files: flat_traces.jsonl and scot_traces.jsonl.
Usage: python data/generate_traces.py --dataset gsm8k --n 2000 --output data/
"""
import argparse, json, re, os, time, random
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

from prompts import TEACHER_SYSTEM, FEW_SHOT_ARITHMETIC

MODE_TAGS = ["abduction","decompose","deduction","induction","analogy","causal"]
# Automatically finds GEMINI_API_KEY from environment 
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def normalize_answer(ans: str) -> str:
    """Normalize extracted answer for robust comparison (strip units, commas, decimals)."""
    # Remove currency symbols and commas
    ans = re.sub(r"[^\d\.]", "", ans)
    # Remove trailing .00
    if ans.endswith(".0"): ans = ans[:-2]
    if ans.endswith(".00"): ans = ans[:-3]
    return ans.strip()

def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else ""

def is_valid_scot(trace: str, ground_truth: str) -> bool:
    """Validate a structured trace against all quality rules."""
    extracted = extract_answer(trace)
    ans_correct = normalize_answer(extracted) == normalize_answer(ground_truth)
    
    has_reasoning = "<reasoning>" in trace and "</reasoning>" in trace
    has_answer    = "<answer>" in trace
    has_meta      = "<meta_reasoning>" in trace
    
    # Mode and Meta sequence validation
    mode_count    = sum(1 for t in MODE_TAGS if f"<{t}>" in trace)
    meta_count    = len(re.findall(r"<meta_reasoning>", trace))
    
    # Substante check (Lite model flexibility)
    meta_blocks = re.findall(r"<meta_reasoning>(.*?)</meta_reasoning>", trace, re.DOTALL)
    meta_substance = all(len(b.split()) >= 6 for b in meta_blocks) if meta_blocks else False

    reasons = []
    if not ans_correct: reasons.append(f"Answer mismatch ('{extracted}' vs '{ground_truth}')")
    if not (has_reasoning and has_answer and has_meta): reasons.append("Missing required tags")
    if mode_count < 2: reasons.append(f"Insufficient modes ({mode_count})")
    if meta_count < mode_count + 1: reasons.append(f"Insufficient meta blocks ({meta_count} for {mode_count} modes)")
    if not meta_substance: reasons.append("Thin meta-reasoning blocks")

    if reasons:
        # Logging failure reasons to terminal for tracking
        # tqdm.write(f"  ❌ Validation Failed: {', '.join(reasons)}")
        return False
    return True

def call_with_backoff(model, prompt, max_retries=10):
    """Call Gemini with exponential backoff to handle rate limits."""
    for i in range(max_retries):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "ResourceExhausted" in err_str:
                delay = (2 ** (i + 1)) + random.uniform(0, 1)
                tqdm.write(f"⚠️ Rate limited. Backing off for {delay:.2f}s...")
                time.sleep(delay)
            else:
                tqdm.write(f"❌ API Error: {err_str}")
                time.sleep(2)
    return None

def generate_scot(problem: str, ground_truth: str, model_name: str, max_attempts: int = 5) -> dict | None:
    """Call teacher model, retry up to max_attempts times."""
    user_msg = FEW_SHOT_ARITHMETIC + f"\n\nProblem: {problem}"
    model = genai.GenerativeModel(model_name, system_instruction=TEACHER_SYSTEM)
    
    for attempt in range(max_attempts):
        resp = call_with_backoff(model, user_msg)
        if not resp: continue
        
        trace = resp.text
        if is_valid_scot(trace, ground_truth):
            modes_used = [t for t in MODE_TAGS if f"<{t}>" in trace]
            meta_texts = re.findall(r"<meta_reasoning>(.*?)</meta_reasoning>", trace, re.DOTALL)
            return {
                "problem":        problem,
                "scot_trace":     trace,
                "answer":         ground_truth,
                "modes_used":     modes_used,
                "meta_count":     len(meta_texts),
                "meta_texts":     [m.strip() for m in meta_texts],
                "mode_sequence":  re.findall(r"<(meta_reasoning|" + "|".join(MODE_TAGS) + r")>", trace),
            }
        elif attempt < max_attempts - 1:
            # We don't log the reason here to keep terminal clean, but we could if needed
            pass
            
    return None

def generate_flat(problem: str, ground_truth: str, model_name: str) -> dict | None:
    """Generate a standard flat <think> trace (ablation baseline)."""
    flat_system = "Solve the problem step by step inside <think>...</think> tags. Give your final answer in <answer>...</answer>."
    model = genai.GenerativeModel(model_name, system_instruction=flat_system)
    
    resp = call_with_backoff(model, problem)
    if not resp: return None
    
    trace = resp.text
    if normalize_answer(extract_answer(trace)) == normalize_answer(ground_truth):
        return {"problem": problem, "flat_trace": trace, "answer": ground_truth}
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k","math","arc"])
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--output", default="data/")
    parser.add_argument("--start_index", type=int, default=None, help="Force start from this index (0-indexed)")
    args = parser.parse_args()

    Path(args.output).mkdir(exist_ok=True, parents=True)

    # Resumption Logic
    scot_path = Path(args.output) / "scot_traces.jsonl"
    flat_path = Path(args.output) / "flat_traces.jsonl"
    
    existing_count = 0
    if scot_path.exists():
        with open(scot_path, "r", encoding="utf-8") as f:
            existing_count = sum(1 for _ in f)
    
    # Use start_index if provided, otherwise auto-resume
    start_from = args.start_index if args.start_index is not None else existing_count

    if start_from >= args.n:
        print(f"✅ Found {existing_count} existing traces in {scot_path}. Goal of {args.n} reached (start_from={start_from}).")
        return

    if args.dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="train")
        problems = [(row["question"], row["answer"].split("####")[-1].strip()) for row in ds]
    elif args.dataset == "arc":
        ds = load_dataset("ai2_arc", "ARC-Challenge", split="train")
        problems = [
            (row["question"], row["choices"]["text"][row["choices"]["label"].index(row["answerKey"])])
            for row in ds
        ]

    print(f"Starting from sample {start_from}/{args.n} (Existing: {existing_count})...")
    
    success, total = existing_count, start_from

    with open(scot_path, "a", encoding="utf-8") as scot_out, \
         open(flat_path, "a", encoding="utf-8") as flat_out:
        
        for problem, answer in tqdm(problems[start_from:args.n], desc=f"Generating {args.dataset}", initial=start_from, total=args.n):
            total += 1
            scot = generate_scot(problem, answer, args.model)
            flat = generate_flat(problem, answer, args.model)
            
            if scot:
                scot_out.write(json.dumps(scot) + "\n")
                scot_out.flush()
                success += 1
            if flat:
                flat_out.write(json.dumps(flat) + "\n")
                flat_out.flush()

    print(f"\nGeneration Complete: {success}/{total} valid S-CoT traces written to {scot_path}")

if __name__ == "__main__":
    main()
