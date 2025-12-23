"""
CLI: select the best candidate by a reference-free verifier.

Scores each candidate against the problem's natural language statement and
selects the max-score candidate.
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable

from .hf_datasets import load_dataset_split
from .verifier import NLVerifier


def _load_nl_map(dataset: str, split: str, id_key: str, nl_key: str) -> dict[str, str]:
    ds = load_dataset_split(dataset, split)
    out: dict[str, str] = {}
    for r in ds:
        if id_key not in r:
            raise ValueError(f"Missing {id_key!r} in dataset row keys: {list(r.keys())}")
        pid = str(r[id_key])
        if pid in out:
            continue
        if nl_key not in r:
            raise ValueError(f"Missing {nl_key!r} in dataset row for id={pid!r}")
        out[pid] = "" if r[nl_key] is None else str(r[nl_key])
    return out


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, action="append", required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--dataset-id-key", type=str, default="id")
    p.add_argument("--dataset-nl-key", type=str, default="nl_statement")

    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--problem-id-key", type=str, default="problem_id")
    p.add_argument("--candidates-key", type=str, default="candidates")
    p.add_argument("--typechecks-key", type=str, default="typechecks")

    p.add_argument("--device", type=str, default="", help="e.g. cuda:0, cuda:1, or cpu (default: auto)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--use-features", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--minimize", action="store_true", help="Select the lowest score instead of the highest.")
    p.add_argument("--emit-scores", action="store_true")
    p.add_argument("--stats-md", type=str, default="", help="Optional markdown summary path")
    p.add_argument("--stats-json", type=str, default="", help="Optional JSON summary path")
    args = p.parse_args()

    nl_map = _load_nl_map(str(args.dataset), str(args.split), str(args.dataset_id_key), str(args.dataset_nl_key))

    model_names = args.model if isinstance(args.model, list) else [args.model]
    verifiers = [
        NLVerifier(
            model_name_or_path=str(name),
            max_length=int(args.max_length),
            device=str(args.device).strip() or None,
            use_features=bool(args.use_features),
        )
        for name in model_names
    ]

    total = 0
    top1_typechecks = 0
    rescued_by_typecheck = 0
    no_typecheck_survivors = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        for obj in _iter_jsonl(str(args.input)):
            if args.problem_id_key not in obj:
                raise ValueError(f"Missing {args.problem_id_key!r} in input row: {obj}")
            pid = str(obj[args.problem_id_key])
            cands = obj.get(args.candidates_key) or []
            if not isinstance(cands, list):
                raise ValueError(f"Expected {args.candidates_key!r} to be a list for problem_id={pid!r}")
            if not cands:
                continue
            if pid not in nl_map:
                raise ValueError(f"Missing nl_statement for problem_id={pid!r}")
            nl = nl_map[pid]

            all_scores: list[list[float]] = []
            for verifier in verifiers:
                all_scores.append(
                    verifier.score_pairs([nl] * len(cands), [str(c) for c in cands], batch_size=int(args.batch_size))
                )
            if not all_scores:
                raise ValueError("No models provided.")
            n = len(all_scores[0])
            if any(len(s) != n for s in all_scores):
                raise ValueError("Score length mismatch across models.")
            scores = [sum(vals) / len(all_scores) for vals in zip(*all_scores)]
            if args.minimize:
                raw_idx = min(range(len(scores)), key=lambda i: scores[i])
            else:
                raw_idx = max(range(len(scores)), key=lambda i: scores[i])

            typechecks = obj.get(args.typechecks_key)
            typecheck_mask = None
            if typechecks is not None:
                if not isinstance(typechecks, list):
                    raise ValueError(f"Expected {args.typechecks_key!r} to be a list for problem_id={pid!r}")
                if len(typechecks) != len(cands):
                    raise ValueError(
                        f"Length mismatch for {pid}: candidates={len(cands)} {args.typechecks_key}={len(typechecks)}"
                    )
                typecheck_mask = [bool(x) for x in typechecks]

            best_idx = raw_idx
            raw_top1_typechecks = None
            no_survivors = False
            if typecheck_mask is not None:
                raw_top1_typechecks = bool(typecheck_mask[raw_idx])
                if any(typecheck_mask):
                    survivors = [i for i, ok in enumerate(typecheck_mask) if ok]
                    if args.minimize:
                        best_idx = min(survivors, key=lambda i: scores[i])
                    else:
                        best_idx = max(survivors, key=lambda i: scores[i])
                    if raw_top1_typechecks is False and typecheck_mask[best_idx]:
                        rescued_by_typecheck += 1
                else:
                    no_survivors = True
                    no_typecheck_survivors += 1

            total += 1
            if raw_top1_typechecks:
                top1_typechecks += 1
            out = {
                "problem_id": pid,
                "chosen_index": int(best_idx),
                "chosen": cands[best_idx],
                "score": float(scores[best_idx]),
            }
            if raw_top1_typechecks is not None:
                out["raw_top1_index"] = int(raw_idx)
                out["raw_top1_typechecks"] = bool(raw_top1_typechecks)
            if no_survivors:
                out["no_typecheck_survivors"] = True
            if args.emit_scores:
                out["scores"] = [float(s) for s in scores]
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    if args.stats_md or args.stats_json:
        stats = {
            "problems": total,
            "top1_typechecks_rate": (top1_typechecks / total) if total else 0.0,
            "top1_typechecks": top1_typechecks,
            "rescued_by_typecheck_filter": rescued_by_typecheck,
            "no_typecheck_survivors": no_typecheck_survivors,
        }
        if args.stats_json:
            with open(args.stats_json, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
                f.write("\n")
        if args.stats_md:
            lines = [
                "# Verifier typecheck stats",
                "",
                f"Problems: {total}",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| top1_typechecks_rate | {100.0 * stats['top1_typechecks_rate']:.1f}% |",
                f"| top1_typechecks | {stats['top1_typechecks']} |",
                f"| rescued_by_typecheck_filter | {stats['rescued_by_typecheck_filter']} |",
                f"| no_typecheck_survivors | {stats['no_typecheck_survivors']} |",
                "",
            ]
            with open(args.stats_md, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))


if __name__ == "__main__":
    main()
