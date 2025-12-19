"""
CLI: analyze BEq+ outcomes vs label correctness for two selection methods.

This is meant to separate:
  - selection quality w.r.t. semantic labels (`correct`)
  - certifiability under BEq+ at a fixed timeout

Inputs:
  - grouped candidates JSONL with `candidates` and `labels`
  - two selections JSONLs (A and B), each with `chosen_index`
  - BEq+ paired results JSONL produced by `beqcritic.paper_pipeline.beq_plus_eval`
    (must include `a_ok` and `b_ok`)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from .features import extract_features
from .textnorm import normalize_lean_statement


def _load_jsonl_map(path: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            pid = obj.get("problem_id")
            if pid is None:
                raise ValueError(f"Missing problem_id in {path}: {obj}")
            out[str(pid)] = obj
    return out


@dataclass(frozen=True)
class _Row:
    pid: str
    a_label_ok: bool
    b_label_ok: bool
    a_beq_ok: bool
    b_beq_ok: bool
    a_feats: dict[str, int]
    b_feats: dict[str, int]


def _mean_int(xs: list[int]) -> float:
    return float(sum(xs)) / max(1, len(xs))


def _summarize_feats(rows: list[_Row], *, which: str) -> dict[str, float]:
    if which not in ["a", "b"]:
        raise ValueError("which must be 'a' or 'b'")
    n_chars: list[int] = []
    n_binders: list[int] = []
    n_prop: list[int] = []
    for r in rows:
        feats = r.a_feats if which == "a" else r.b_feats
        n_chars.append(int(feats["n_chars"]))
        n_binders.append(int(feats["n_binders"]))
        n_prop.append(int(feats["n_prop_assumptions"]))
    return {
        "n": float(len(rows)),
        "avg_chars": _mean_int(n_chars),
        "avg_binders": _mean_int(n_binders),
        "avg_prop_assumptions": _mean_int(n_prop),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=str, required=True, help="Grouped candidates JSONL (must include `labels`)")
    p.add_argument("--selections-a", type=str, required=True)
    p.add_argument("--selections-b", type=str, required=True)
    p.add_argument("--beqplus-results", type=str, required=True, help="JSONL from beqcritic.paper_pipeline.beq_plus_eval")
    p.add_argument("--name-a", type=str, default="A")
    p.add_argument("--name-b", type=str, default="B")
    p.add_argument("--output-jsonl", type=str, default="", help="Optional joined per-problem JSONL")
    p.add_argument("--max-examples", type=int, default=20, help="Max problem_ids printed per category")
    args = p.parse_args()

    cand = _load_jsonl_map(args.candidates)
    sel_a = _load_jsonl_map(args.selections_a)
    sel_b = _load_jsonl_map(args.selections_b)
    res = _load_jsonl_map(args.beqplus_results)

    # Expect paired results.
    sample = next(iter(res.values()), None)
    if sample is None:
        raise SystemExit("Empty --beqplus-results")
    if "a_ok" not in sample or "b_ok" not in sample:
        raise ValueError("--beqplus-results must contain paired fields `a_ok` and `b_ok`")

    pids = sorted(set(cand) & set(sel_a) & set(sel_b) & set(res))
    if not pids:
        raise SystemExit("No overlapping problem_ids across candidates/selections/results.")

    missing = {
        "candidates": len(set(sel_a) - set(cand)),
        "selections_a": len(set(cand) - set(sel_a)),
        "selections_b": len(set(cand) - set(sel_b)),
        "beqplus_results": len(set(cand) - set(res)),
    }
    for k, v in missing.items():
        if v:
            print(f"Warning: {v} ids missing {k}")

    rows: list[_Row] = []
    out_f = open(args.output_jsonl, "w", encoding="utf-8") if str(args.output_jsonl).strip() else None
    try:
        for pid in pids:
            c = cand[pid]
            candidates = c.get("candidates") or []
            labels = c.get("labels") or []
            if not candidates:
                continue
            if not isinstance(labels, list) or len(labels) != len(candidates):
                raise ValueError(f"Candidates/labels length mismatch for {pid}: {len(candidates)} vs {len(labels)}")
            labels01 = [1 if int(x) else 0 for x in labels]

            a_idx = int(sel_a[pid].get("chosen_index"))
            b_idx = int(sel_b[pid].get("chosen_index"))
            if a_idx < 0 or a_idx >= len(labels01):
                raise ValueError(f"A chosen_index out of range for {pid}: {a_idx} (n={len(labels01)})")
            if b_idx < 0 or b_idx >= len(labels01):
                raise ValueError(f"B chosen_index out of range for {pid}: {b_idx} (n={len(labels01)})")

            a_stmt = str(candidates[a_idx])
            b_stmt = str(candidates[b_idx])
            a_norm = normalize_lean_statement(a_stmt)
            b_norm = normalize_lean_statement(b_stmt)
            a_f = extract_features(a_norm)
            b_f = extract_features(b_norm)

            a_label_ok = bool(labels01[a_idx])
            b_label_ok = bool(labels01[b_idx])
            a_beq_ok = bool(res[pid].get("a_ok"))
            b_beq_ok = bool(res[pid].get("b_ok"))

            row = _Row(
                pid=str(pid),
                a_label_ok=bool(a_label_ok),
                b_label_ok=bool(b_label_ok),
                a_beq_ok=bool(a_beq_ok),
                b_beq_ok=bool(b_beq_ok),
                a_feats={
                    "n_chars": int(a_f.n_chars),
                    "n_binders": int(a_f.n_binders),
                    "n_prop_assumptions": int(a_f.n_prop_assumptions),
                },
                b_feats={
                    "n_chars": int(b_f.n_chars),
                    "n_binders": int(b_f.n_binders),
                    "n_prop_assumptions": int(b_f.n_prop_assumptions),
                },
            )
            rows.append(row)

            if out_f is not None:
                out_f.write(
                    json.dumps(
                        {
                            "problem_id": str(pid),
                            "n_candidates": int(len(candidates)),
                            "has_any_correct": bool(any(labels01)),
                            "a": {
                                "chosen_index": int(a_idx),
                                "label_ok": bool(a_label_ok),
                                "beq_ok": bool(a_beq_ok),
                                "features": row.a_feats,
                            },
                            "b": {
                                "chosen_index": int(b_idx),
                                "label_ok": bool(b_label_ok),
                                "beq_ok": bool(b_beq_ok),
                                "features": row.b_feats,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if out_f is not None:
            out_f.close()

    def _count(pred) -> int:
        return sum(1 for r in rows if pred(r))

    n = len(rows)
    a_lab = _count(lambda r: r.a_label_ok)
    b_lab = _count(lambda r: r.b_label_ok)
    a_beq = _count(lambda r: r.a_beq_ok)
    b_beq = _count(lambda r: r.b_beq_ok)

    def _pct(x: int, d: int) -> float:
        return 100.0 * x / max(1, d)

    print(f"Problems: {n}")
    print(f"{args.name_a} label-correct: {a_lab} ({_pct(a_lab, n):.1f}%)")
    print(f"{args.name_b} label-correct: {b_lab} ({_pct(b_lab, n):.1f}%)")
    print(f"{args.name_a} BEq+ ok: {a_beq} ({_pct(a_beq, n):.1f}%)")
    print(f"{args.name_b} BEq+ ok: {b_beq} ({_pct(b_beq, n):.1f}%)")

    a_cert = _count(lambda r: r.a_label_ok and r.a_beq_ok)
    b_cert = _count(lambda r: r.b_label_ok and r.b_beq_ok)
    print(f"{args.name_a} BEq+ ok | label-correct: {a_cert}/{a_lab} ({_pct(a_cert, a_lab):.1f}%)")
    print(f"{args.name_b} BEq+ ok | label-correct: {b_cert}/{b_lab} ({_pct(b_cert, b_lab):.1f}%)")

    both_label_ok = [r for r in rows if r.a_label_ok and r.b_label_ok]
    both_label_ok_a_only = [r for r in both_label_ok if r.a_beq_ok and not r.b_beq_ok]
    both_label_ok_b_only = [r for r in both_label_ok if r.b_beq_ok and not r.a_beq_ok]
    print()
    print(f"Both label-correct: {len(both_label_ok)}")
    print(f"BEq+ differs within label-correct subset: {args.name_a} only={len(both_label_ok_a_only)}, {args.name_b} only={len(both_label_ok_b_only)}")

    a_lc_beq_ok = [r for r in rows if r.a_label_ok and r.a_beq_ok]
    a_lc_beq_fail = [r for r in rows if r.a_label_ok and not r.a_beq_ok]
    b_lc_beq_ok = [r for r in rows if r.b_label_ok and r.b_beq_ok]
    b_lc_beq_fail = [r for r in rows if r.b_label_ok and not r.b_beq_ok]

    print()
    print(f"{args.name_a} label-correct & BEq+ ok feature means: {_summarize_feats(a_lc_beq_ok, which='a')}")
    print(f"{args.name_a} label-correct & BEq+ fail feature means: {_summarize_feats(a_lc_beq_fail, which='a')}")
    print(f"{args.name_b} label-correct & BEq+ ok feature means: {_summarize_feats(b_lc_beq_ok, which='b')}")
    print(f"{args.name_b} label-correct & BEq+ fail feature means: {_summarize_feats(b_lc_beq_fail, which='b')}")

    k = max(0, int(args.max_examples))
    if k:
        print()
        print(f"Examples where both label-correct but {args.name_a} BEq+ ok and {args.name_b} BEq+ fails:")
        for r in both_label_ok_a_only[:k]:
            print(f"  {r.pid}")
        print()
        print(f"Examples where both label-correct but {args.name_b} BEq+ ok and {args.name_a} BEq+ fails:")
        for r in both_label_ok_b_only[:k]:
            print(f"  {r.pid}")


if __name__ == "__main__":
    main()

