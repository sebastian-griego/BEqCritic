"""
CLI: group row-level candidate JSONL into BEqCritic's grouped-candidates JSONL.

This is the adapter you want when another pipeline (e.g. the paper's) produces a
"flat" file with one candidate per line.

Input JSONL (one candidate per line):
  {"problem_id": "...", "candidate": "...", "label": 0, "reference": "..."}

Output JSONL (one problem per line):
  {"problem_id": "...", "candidates": ["...", ...], "labels": [0, 1, ...], "reference": "..."}

Only `problem_id` and `candidates` are required by `beqcritic.score_and_select`.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from .labels import coerce_binary_label


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Flat JSONL (one candidate per line)")
    p.add_argument("--output", type=str, required=True, help="Grouped JSONL (one problem per line)")
    p.add_argument("--problem-id-key", type=str, default="problem_id")
    p.add_argument("--candidate-key", type=str, default="candidate")
    p.add_argument("--label-key", type=str, default="", help="Optional label key (writes `labels` list)")
    p.add_argument("--reference-key", type=str, default="", help="Optional reference key (writes `reference`)")
    p.add_argument(
        "--allow-reference-mismatch",
        action="store_true",
        help="If set, keep the first reference per problem_id even if later rows disagree.",
    )
    p.add_argument("--dedupe", action="store_true", help="Drop duplicate candidate strings per problem")
    p.add_argument("--drop-empty", action="store_true", help="Drop candidates that are empty/whitespace")
    p.add_argument("--sort-by-problem-id", action="store_true", help="Sort groups by problem_id before writing")
    p.add_argument("--max-problems", type=int, default=0, help="Limit number of problems (0 = no limit)")
    args = p.parse_args()

    grouped: dict[str, dict[str, Any]] = {}
    seen: dict[str, set[str]] = {}

    n_rows = 0
    n_kept_rows = 0
    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            n_rows += 1
            obj = json.loads(line)
            if args.problem_id_key not in obj:
                raise ValueError(f"Missing {args.problem_id_key!r} in input row: {obj}")
            pid = str(obj[args.problem_id_key])

            if args.max_problems and pid not in grouped and len(grouped) >= int(args.max_problems):
                break

            cand_raw = obj.get(args.candidate_key)
            cand = "" if cand_raw is None else str(cand_raw)
            if args.drop_empty and not cand.strip():
                continue

            if pid not in grouped:
                grouped[pid] = {"candidates": []}
                if args.label_key:
                    grouped[pid]["labels"] = []
                if args.reference_key:
                    if args.reference_key not in obj:
                        raise ValueError(f"Missing {args.reference_key!r} in input row: {obj}")
                    grouped[pid]["reference"] = obj.get(args.reference_key)
                seen[pid] = set()

            if args.dedupe:
                if cand in seen[pid]:
                    continue
                seen[pid].add(cand)

            grouped[pid]["candidates"].append(cand)
            if args.label_key:
                if args.label_key not in obj:
                    raise ValueError(f"Missing {args.label_key!r} in input row: {obj}")
                grouped[pid]["labels"].append(coerce_binary_label(obj.get(args.label_key)))

            if args.reference_key:
                ref = obj.get(args.reference_key)
                if grouped[pid].get("reference") != ref and not args.allow_reference_mismatch:
                    raise ValueError(
                        f"Reference mismatch for problem_id={pid!r}. "
                        f"First={grouped[pid].get('reference')!r}, later={ref!r}. "
                        "Use --allow-reference-mismatch to keep the first."
                    )

            n_kept_rows += 1

    keys = sorted(grouped.keys()) if args.sort_by_problem_id else list(grouped.keys())
    with open(args.output, "w", encoding="utf-8") as fout:
        for pid in keys:
            fout.write(json.dumps({"problem_id": pid, **grouped[pid]}, ensure_ascii=False) + "\n")

    print(f"Read {n_rows} rows, kept {n_kept_rows}, wrote {len(keys)} problems to {args.output}")


if __name__ == "__main__":
    main()
