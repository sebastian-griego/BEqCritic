"""
CLI: cluster candidates by learned equivalence and select a representative.

Input JSONL format (one problem per line):
  {"problem_id": "...", "candidates": ["theorem ...", "...", ...]}

Output JSONL format:
  {"problem_id": "...", "chosen": "...", "chosen_index": 3, "component_size": 7, "component_indices": [..]}
"""
from __future__ import annotations

import argparse
import json

from .modeling import BeqCritic
from .select import select_by_equivalence_clustering

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--tie-break", type=str, default="shortest", choices=["shortest", "first"])
    args = p.parse_args()

    critic = BeqCritic(model_name_or_path=args.model, max_length=args.max_length)

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            candidates = obj.get("candidates", [])
            res = select_by_equivalence_clustering(
                candidates=candidates,
                critic=critic,
                threshold=args.threshold,
                batch_size=args.batch_size,
                tie_break=args.tie_break,
            )
            out = {
                "problem_id": obj.get("problem_id"),
                "chosen": res.chosen_statement,
                "chosen_index": res.chosen_index,
                "component_size": res.component_size,
                "component_indices": res.component_indices,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
