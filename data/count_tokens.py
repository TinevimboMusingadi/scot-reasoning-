import os, json
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Use Gemini 1.5 Pro as requested for high-precision token auditing
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = 'gemini-2.5-pro'

def count_tokens_in_jsonl(file_path: Path):
    if not file_path.exists():
        print(f"⚠️ File not found: {file_path}")
        return 0, 0
    
    model = genai.GenerativeModel(MODEL_NAME)
    total_tokens = 0
    sample_count = 0
    
    print(f"📊 Auditing {file_path.name}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Load all lines into memory for batch counting if possible,
        # but for very large files, line-by-line is safer.
        # However, Gemini count_tokens is fast enough for small batches.
        lines = f.readlines()
        sample_count = len(lines)
        
        # We'll batch in chunks of 50 to avoid hitting API rate limits too hard 
        # while keeping the overhead low.
        batch_size = 50
        for i in tqdm(range(0, len(lines), batch_size), desc="Counting Tokens"):
            batch = lines[i:i+batch_size]
            # Concatenate the text we care about
            batch_text = ""
            for line in batch:
                data = json.loads(line)
                # Count the primary reasoning field
                if "scot_trace" in data:
                    batch_text += data["scot_trace"] + "\n"
                elif "flat_trace" in data:
                    batch_text += data["flat_trace"] + "\n"
                elif "question" in data: # Fallback for original GSM
                    batch_text += data["question"] + "\n"
            
            if batch_text:
                try:
                    # SDK v1 style
                    resp = model.count_tokens(batch_text)
                    total_tokens += resp.total_tokens
                except Exception as e:
                    print(f"Error counting tokens: {e}")
                    
    return total_tokens, sample_count

def main():
    base_dir = Path("data/full_run")
    scot_file = base_dir / "scot_traces.jsonl"
    flat_file = base_dir / "flat_traces.jsonl"
    gsm_file = Path("data/gsm/train.jsonl") # Also check the original for comparison

    results = []
    for f in [scot_file, flat_file, gsm_file]:
        tokens, count = count_tokens_in_jsonl(f)
        if count > 0:
            results.append({
                "file": f.name,
                "samples": count,
                "total_tokens": tokens,
                "avg_per_sample": tokens / count if count > 0 else 0
            })

    print("\n" + "="*40)
    print("       TOKEN AUDIT REPORT")
    print("="*40)
    for res in results:
        print(f"File: {res['file']}")
        print(f"  Samples: {res['samples']}")
        print(f"  Total Tokens: {res['total_tokens']:,}")
        print(f"  Avg Tokens/Sample: {res['avg_per_sample']:.1f}")
        print("-" * 20)
    print("="*40)

if __name__ == "__main__":
    main()
