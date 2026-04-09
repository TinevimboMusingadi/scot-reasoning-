import argparse, json, re, os, time, random
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

from prompts import TEACHER_SYSTEM, FEW_SHOT_ARITHMETIC

MODE_TAGS = ["abduction","decompose","deduction","induction","analogy","causal"]
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else ""

def is_valid_scot_diagnostic(trace: str, ground_truth: str) -> bool:
    reasons = []
    ans_extracted = extract_answer(trace)
    if ans_extracted.lower() != ground_truth.lower():
        reasons.append(f"Answer mismatch: '{ans_extracted}' vs '{ground_truth}'")
    
    if "<reasoning>" not in trace or "</reasoning>" not in trace:
        reasons.append("Missing <reasoning> tags")
    
    if "<answer>" not in trace:
        reasons.append("Missing <answer> tag")
        
    if "<meta_reasoning>" not in trace:
        reasons.append("Missing <meta_reasoning> tag")
        
    mode_count = sum(1 for t in MODE_TAGS if f"<{t}>" in trace)
    if mode_count < 2:
        reasons.append(f"Incomplete structure: Only {mode_count} modes used (need >=2)")
        
    meta_count = len(re.findall(r"<meta_reasoning>", trace))
    if meta_count < mode_count + 1:
        reasons.append(f"Insufficient reflection: {meta_count} meta blocks for {mode_count} modes")
        
    if trace.find("<meta_reasoning>") > 300:
        reasons.append(f"Slow start: First meta-reasoning block starts at char {trace.find('<meta_reasoning>')}")

    meta_blocks = re.findall(r"<meta_reasoning>(.*?)</meta_reasoning>", trace, re.DOTALL)
    for i, b in enumerate(meta_blocks):
        words = len(b.split())
        if words < 10:
            reasons.append(f"Thin meta block {i+1}: Only {words} words")

    if reasons:
        print(f"\n[DIAGNOSTIC] FAILED: {reasons}")
        return False
    return True

def debug_run(n=5):
    ds = load_dataset("gsm8k", "main", split="train")
    problems = [(row["question"], row["answer"].split("####")[-1].strip()) for row in ds]
    model = genai.GenerativeModel('gemini-3.1-flash-lite-preview', system_instruction=TEACHER_SYSTEM)
    
    for problem, answer in problems[:n]:
        print(f"\n--- Prob: {problem[:50]}... ---")
        user_msg = FEW_SHOT_ARITHMETIC + f"\n\nProblem: {problem}"
        try:
            resp = model.generate_content(user_msg)
            is_valid_scot_diagnostic(resp.text, answer)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    debug_run(10)
