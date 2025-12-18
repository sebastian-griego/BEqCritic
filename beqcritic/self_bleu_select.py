"""
CLI: select a single candidate per problem using a Self-BLEU style medoid.

This matches the "Self-BLEU over candidate pairs" selection heuristic used in the
paper pipeline: compute BLEU similarity for all candidate pairs, then pick the
candidate with the best aggregate BLEU (global medoid).

Input JSONL (one problem per line):
  {"problem_id": "...", "candidates": ["...", ...]}

Output JSONL format is compatible with `beqcritic.evaluate_selection`:
  {"problem_id": "...", "chosen": "...", "chosen_index": 3, ...}
"""

from __future__ import annotations

import argparse
import json

from .bleu import bleu_score_matrix
from .select import global_medoid_index


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--bleu-max-n", type=int, default=4)
    p.add_argument("--bleu-smooth", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=1, help="If >1, emit chosen_indices (BLEU centrality ranking).")
    p.add_argument(
        "--emit-topk-text",
        action="store_true",
        help="When used with --top-k > 1, also include the selected top-k Lean strings in output JSONL.",
    )
    args = p.parse_args()

    k = max(1, int(args.top_k))

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = obj.get("problem_id")
            candidates = obj.get("candidates") or []
            if not candidates:
                continue

            norm, scores = bleu_score_matrix(
                candidates=list(candidates),
                max_n=int(args.bleu_max_n),
                smooth=float(args.bleu_smooth),
            )
            chosen_index, chosen_centrality = global_medoid_index(norm=norm, scores=scores)

            chosen_indices = None
            if k > 1:
                n = len(candidates)
                cents = []
                for i in range(n):
                    vals = [scores[i][j] for j in range(n) if j != i]
                    cent = sum(vals) / max(1, len(vals))
                    cents.append(float(cent))
                ranked = sorted(
                    range(n),
                    key=lambda i: (-cents[i], len(norm[i]), i),
                )
                chosen_indices = ranked[: min(k, n)]

            out = {
                "problem_id": pid,
                "chosen": str(candidates[int(chosen_index)]),
                "chosen_index": int(chosen_index),
                "component_size": int(len(candidates)),
                "component_indices": list(range(len(candidates))),
                "selection_method": "bleu_medoid",
                "similarity": "bleu_medoid",
                "chosen_centrality": float(chosen_centrality),
            }
            if int(args.bleu_max_n) != 4 or float(args.bleu_smooth) != 1.0:
                out["bleu_max_n"] = int(args.bleu_max_n)
                out["bleu_smooth"] = float(args.bleu_smooth)
            if chosen_indices is not None:
                out["chosen_indices"] = list(chosen_indices)
                if args.emit_topk_text:
                    out["chosen_topk"] = [str(candidates[int(i)]) for i in chosen_indices]
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

