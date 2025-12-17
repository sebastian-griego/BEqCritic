"""
CLI: build a grouped candidates JSONL file from a row-level HuggingFace dataset split.

Output JSONL (one problem per line):
  {"problem_id": "...", "candidates": ["...", ...], "labels": [0, 1, ...]}

Extra fields are allowed by `beqcritic.score_and_select` (it only reads `problem_id` and `candidates`).
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from .hf_datasets import load_dataset_split
from .textnorm import normalize_lean_statement


def _guess_bool_label(x: Any) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return int(x)
    if isinstance(x, str):
        xl = x.strip().lower()
        if xl in ["1", "true", "yes", "correct", "ok"]:
            return 1
        if xl in ["0", "false", "no", "incorrect", "wrong"]:
            return 0
    raise ValueError(f"Cannot interpret label value: {x!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="valid")
    p.add_argument("--pred-key", type=str, default="prediction")
    p.add_argument("--ref-key", type=str, default="", help="Optional reference key (for length bucketing)")
    p.add_argument("--label-key", type=str, default="label")
    p.add_argument("--problem-id-key", type=str, default="problem_id")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--max-problems", type=int, default=0, help="Limit number of problems (0 = no limit)")
    p.add_argument("--dedupe", action="store_true", help="Drop duplicate candidate strings per problem")
    p.add_argument("--include-reference", action="store_true", help="Include the reference statement text in output")
    args = p.parse_args()

    ds = load_dataset_split(args.dataset, args.split)

    grouped: dict[str, dict[str, Any]] = {}
    seen: dict[str, set[str]] = {}
    for r in ds:
        pid = str(r.get(args.problem_id_key))
        cand = "" if r.get(args.pred_key) is None else str(r.get(args.pred_key))
        lab = _guess_bool_label(r.get(args.label_key))

        if pid not in grouped:
            if args.max_problems and len(grouped) >= int(args.max_problems):
                break
            grouped[pid] = {"candidates": [], "labels": []}
            if args.ref_key:
                ref = "" if r.get(args.ref_key) is None else str(r.get(args.ref_key))
                norm_ref = normalize_lean_statement(ref)
                grouped[pid]["ref_len_chars"] = len(norm_ref)
                if args.include_reference:
                    grouped[pid]["reference"] = ref
            seen[pid] = set()

        if args.dedupe:
            if cand in seen[pid]:
                continue
            seen[pid].add(cand)

        grouped[pid]["candidates"].append(cand)
        grouped[pid]["labels"].append(lab)

    with open(args.output, "w", encoding="utf-8") as f:
        for pid, obj in grouped.items():
            out = {"problem_id": pid, **obj}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
