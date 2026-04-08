"""
Quick stats check on generated traces before training.
Usage: python data/validate_traces.py --file data/scot_traces.jsonl
"""
import json, sys, re, argparse
from collections import Counter

MODE_TAGS = ["abduction","decompose","deduction","induction","analogy","causal"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    records = [json.loads(l) for l in open(args.file)]
    print(f"Total records: {len(records)}")

    mode_counts = Counter()
    meta_counts = []
    for r in records:
        for m in r.get("modes_used", []):
            mode_counts[m] += 1
        meta_counts.append(r.get("meta_count", 0))

    if not mode_counts:
        print("No modes recorded.")
        return

    print("\nMode distribution:")
    for m, c in mode_counts.most_common():
        print(f"  {m:15s}: {c} ({100*c/len(records):.1f}%)")

    print(f"\nAvg meta_reasoning blocks per trace: {sum(meta_counts)/max(len(meta_counts), 1):.2f}")
    print(f"Traces with >= 3 meta blocks: {sum(1 for x in meta_counts if x >= 3)}")

if __name__ == "__main__":
    main()
