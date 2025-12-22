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
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--dataset-id-key", type=str, default="id")
    p.add_argument("--dataset-nl-key", type=str, default="nl_statement")

    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--problem-id-key", type=str, default="problem_id")
    p.add_argument("--candidates-key", type=str, default="candidates")

    p.add_argument("--device", type=str, default="", help="e.g. cuda:0, cuda:1, or cpu (default: auto)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--use-features", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--emit-scores", action="store_true")
    args = p.parse_args()

    nl_map = _load_nl_map(str(args.dataset), str(args.split), str(args.dataset_id_key), str(args.dataset_nl_key))

    verifier = NLVerifier(
        model_name_or_path=str(args.model),
        max_length=int(args.max_length),
        device=str(args.device).strip() or None,
        use_features=bool(args.use_features),
    )

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

            scores = verifier.score_pairs([nl] * len(cands), [str(c) for c in cands], batch_size=int(args.batch_size))
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            out = {
                "problem_id": pid,
                "chosen_index": int(best_idx),
                "chosen": cands[best_idx],
                "score": float(scores[best_idx]),
            }
            if args.emit_scores:
                out["scores"] = [float(s) for s in scores]
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
