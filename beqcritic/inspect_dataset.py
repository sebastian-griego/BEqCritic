"""
Small helper to print the column names of a HuggingFace dataset split.

Example:
  python -m beq_critic.inspect_dataset --dataset PAug/ProofNetVerif --split train
"""
from __future__ import annotations
import argparse
from datasets import load_dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--n", type=int, default=2, help="Number of rows to print")
    args = p.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    print("Columns:", ds.column_names)
    for i in range(min(args.n, len(ds))):
        row = ds[i]
        keys = list(row.keys())
        print(f"Row[{i}] keys:", keys)
        for k in keys[:12]:
            v = row[k]
            s = str(v)
            if len(s) > 160:
                s = s[:160] + "..."
            print(" ", k, "=", s)

if __name__ == "__main__":
    main()
