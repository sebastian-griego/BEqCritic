"""
CLI: compare two selection outputs on the same grouped-candidates file.

This is a lightweight A/B evaluator for the "selected correct" metric used by
`beqcritic.evaluate_selection`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class _Problem:
    any_correct: bool
    a_correct: bool
    b_correct: bool
    n_candidates: int


def _load_jsonl_map(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = obj.get("problem_id")
            if pid is None:
                raise ValueError(f"Missing problem_id in {path}: {obj}")
            out[str(pid)] = obj
    return out


def _parse_timing(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                out[k] = float(v)
            except Exception:
                continue
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=str, required=True)
    p.add_argument("--selections-a", type=str, required=True)
    p.add_argument("--selections-b", type=str, required=True)
    p.add_argument("--a-name", type=str, default="A")
    p.add_argument("--b-name", type=str, default="B")
    p.add_argument("--max-problems", type=int, default=0, help="Limit number of problems (0 = no limit).")
    p.add_argument(
        "--timing",
        type=str,
        default="",
        help="Optional timing.txt (as produced by scripts/run_quickstart.sh) to include rough wall times.",
    )
    p.add_argument("--output-json", type=str, default="", help="Optional path to write metrics JSON.")
    p.add_argument("--output-md", type=str, default="", help="Optional path to write a Markdown summary.")
    args = p.parse_args()

    cand = _load_jsonl_map(args.candidates)
    sel_a = _load_jsonl_map(args.selections_a)
    sel_b = _load_jsonl_map(args.selections_b)

    pids = sorted(set(cand.keys()) & set(sel_a.keys()) & set(sel_b.keys()))
    if int(args.max_problems) > 0:
        pids = pids[: int(args.max_problems)]
    if not pids:
        raise SystemExit("No overlapping problem_ids across candidates and both selection files.")

    probs: list[_Problem] = []
    total_pairs = 0
    total_candidates = 0
    for pid in pids:
        c = cand[pid]
        candidates = c.get("candidates") or []
        labels = c.get("labels") or []
        if len(candidates) != len(labels):
            raise ValueError(f"Candidates/labels length mismatch for {pid}: {len(candidates)} vs {len(labels)}")
        if not candidates:
            continue

        n = len(candidates)
        total_candidates += n
        total_pairs += n * (n - 1) // 2

        labels01 = [1 if int(x) else 0 for x in labels]
        any_correct = any(labels01)

        ia = int(sel_a[pid].get("chosen_index"))
        ib = int(sel_b[pid].get("chosen_index"))
        if ia < 0 or ia >= n:
            raise ValueError(f"{args.a_name} chosen_index out of range for {pid}: {ia} (n={n})")
        if ib < 0 or ib >= n:
            raise ValueError(f"{args.b_name} chosen_index out of range for {pid}: {ib} (n={n})")

        probs.append(
            _Problem(
                any_correct=bool(any_correct),
                a_correct=bool(labels01[ia]),
                b_correct=bool(labels01[ib]),
                n_candidates=n,
            )
        )

    if not probs:
        raise SystemExit("No non-empty candidate sets found.")

    n = len(probs)
    any_mask = [p.any_correct for p in probs]
    n_any = sum(any_mask)

    a_vals = [1.0 if p.a_correct else 0.0 for p in probs]
    b_vals = [1.0 if p.b_correct else 0.0 for p in probs]
    diff_vals = [b - a for a, b in zip(a_vals, b_vals)]

    a_mean = sum(a_vals) / n
    b_mean = sum(b_vals) / n
    diff_mean = sum(diff_vals) / n

    a_any_vals = [1.0 if p.a_correct else 0.0 for p in probs if p.any_correct]
    b_any_vals = [1.0 if p.b_correct else 0.0 for p in probs if p.any_correct]
    a_any_mean = (sum(a_any_vals) / max(1, len(a_any_vals))) if a_any_vals else 0.0
    b_any_mean = (sum(b_any_vals) / max(1, len(b_any_vals))) if b_any_vals else 0.0

    timing = _parse_timing(args.timing) if str(args.timing).strip() else {}
    a_time_s = timing.get("select_selfbleu_seconds") if str(args.a_name).lower() == "selfbleu" else None
    b_time_s = timing.get("select_beqcritic_seconds") if str(args.b_name).lower() == "beqcritic" else None
    # Generic fallback if the caller didn't use our quickstart naming.
    if a_time_s is None:
        a_time_s = timing.get(f"select_{str(args.a_name).strip()}_seconds")
    if b_time_s is None:
        b_time_s = timing.get(f"select_{str(args.b_name).strip()}_seconds")

    out = {
        "dataset_metrics": {
            "problems": int(n),
            "has_any_correct": int(n_any),
            "has_any_correct_pct": 100.0 * n_any / max(1, n),
            "avg_candidates_per_problem": float(total_candidates) / float(max(1, n)),
            "total_candidate_pairs": int(total_pairs),
        },
        "a": {
            "name": str(args.a_name),
            "selected_correct_pct": 100.0 * float(a_mean),
            "selected_correct_given_any_pct": 100.0 * float(a_any_mean),
            "selection_time_seconds": float(a_time_s) if a_time_s is not None else None,
        },
        "b": {
            "name": str(args.b_name),
            "selected_correct_pct": 100.0 * float(b_mean),
            "selected_correct_given_any_pct": 100.0 * float(b_any_mean),
            "selection_time_seconds": float(b_time_s) if b_time_s is not None else None,
        },
        "b_minus_a": {
            "selected_correct_pct": 100.0 * float(diff_mean),
        },
        "timing_seconds": timing,
    }

    md = []
    md.append(f"# A/B selection comparison ({args.a_name} vs {args.b_name})")
    md.append("")
    md.append(f"- Problems: {n} (any-correct: {n_any}, {100.0*n_any/max(1,n):.1f}%)")
    md.append(f"- Avg candidates/problem: {float(total_candidates)/float(max(1,n)):.2f}")
    md.append(f"- Total candidate pairs (n(n-1)/2 summed): {total_pairs}")
    md.append("")
    md.append("| method | selected correct (%) | selected correct \\| any correct (%) | selection time (s) | pairwise comps |")
    md.append("|---|---:|---:|---:|---:|")
    a_time_str = f"{float(a_time_s):.0f}" if a_time_s is not None else "-"
    b_time_str = f"{float(b_time_s):.0f}" if b_time_s is not None else "-"
    md.append(
        f"| {args.a_name} | {100.0*a_mean:.1f} | {100.0*a_any_mean:.1f} | {a_time_str} | {total_pairs} |"
    )
    md.append(
        f"| {args.b_name} | {100.0*b_mean:.1f} | {100.0*b_any_mean:.1f} | {b_time_str} | {total_pairs} |"
    )
    md.append(
        f"| {args.b_name} - {args.a_name} | {100.0*diff_mean:+.1f} | - | - | - |"
    )
    md.append("")
    md_text = "\n".join(md) + "\n"

    print(md_text, end="")

    if str(args.output_md).strip():
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md_text)
    if str(args.output_json).strip():
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
