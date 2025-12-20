#!/usr/bin/env python3
"""
Summarize a selection JSONL against grouped-candidates labels.

Inputs:
  - candidates JSONL (one problem per line) from beqcritic.make_grouped_candidates
    (must include `labels`)
  - selections JSONL from beqcritic.score_and_select or beqcritic.self_bleu_select
    (must include `chosen_index` or `chosen_indices`)
"""

from __future__ import annotations

import argparse
import json


def _load_jsonl_map(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = obj.get("problem_id")
            if pid is None:
                raise ValueError(f"Missing problem_id at {path}:{line_no}")
            out[str(pid)] = obj
    return out


def _normalize_lean_statement(s: str) -> str:
    # Import lazily so this script works even when run without an editable install.
    from beqcritic.textnorm import normalize_lean_statement

    return normalize_lean_statement(s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--selections", required=True)
    p.add_argument("--name", default="")
    p.add_argument("--max-k", type=int, default=1)
    p.add_argument("--output-json", default="")
    args = p.parse_args()

    cand = _load_jsonl_map(str(args.candidates))
    sel = _load_jsonl_map(str(args.selections))

    pids = sorted(set(cand.keys()) & set(sel.keys()))
    if not pids:
        raise SystemExit("No overlapping problem_ids across candidates and selections.")

    max_k = max(1, int(args.max_k))
    selected_hits: list[float] = []
    selected_hits_any: list[float] = []
    topk_hits: list[list[float]] = [[] for _ in range(max_k)]
    topk_hits_any: list[list[float]] = [[] for _ in range(max_k)]

    n = 0
    n_any = 0
    total_candidates = 0
    first_hits = 0
    shortest_hits = 0

    for pid in pids:
        c = cand[pid]
        s = sel[pid]

        candidates = c.get("candidates") or []
        labels = c.get("labels") or []
        if len(candidates) != len(labels):
            raise ValueError(f"Candidates/labels length mismatch for {pid}: {len(candidates)} vs {len(labels)}")
        if not candidates:
            continue

        chosen_indices_raw = s.get("chosen_indices", None)
        if isinstance(chosen_indices_raw, list) and chosen_indices_raw:
            chosen_indices = [int(x) for x in chosen_indices_raw]
        else:
            if "chosen_index" not in s:
                raise ValueError(f"Missing chosen_index in selections for {pid}")
            chosen_indices = [int(s.get("chosen_index"))]

        bad = [i for i in chosen_indices if i < 0 or i >= len(labels)]
        if bad:
            bad.sort()
            raise ValueError(f"chosen_indices out of range for {pid}: {bad[:5]} (n={len(labels)})")

        labels01 = [1 if int(x) else 0 for x in labels]
        any_correct = any(labels01)
        sel_correct = bool(labels01[int(chosen_indices[0])])

        n += 1
        total_candidates += len(candidates)
        n_any += int(any_correct)
        selected_hits.append(1.0 if sel_correct else 0.0)
        if any_correct:
            selected_hits_any.append(1.0 if sel_correct else 0.0)

        for k in range(1, max_k + 1):
            picked = chosen_indices[:k]
            hit = any(bool(labels01[i]) for i in picked)
            topk_hits[k - 1].append(1.0 if hit else 0.0)
            if any_correct:
                topk_hits_any[k - 1].append(1.0 if hit else 0.0)

        first_hits += int(bool(labels01[0]))
        norm = [_normalize_lean_statement(x) for x in candidates]
        shortest_idx = min(range(len(norm)), key=lambda i: (len(norm[i]), i))
        shortest_hits += int(bool(labels01[shortest_idx]))

    if n == 0:
        raise SystemExit("No non-empty candidate sets found.")

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    out: dict[str, object] = {
        "name": str(args.name).strip() or None,
        "problems": int(n),
        "has_any_correct": int(n_any),
        "has_any_correct_pct": 100.0 * float(n_any) / max(1, n),
        "avg_candidates_per_problem": float(total_candidates) / max(1, n),
        "selected_correct_pct": 100.0 * _mean(selected_hits),
        "selected_correct_given_any_pct": 100.0 * (_mean(selected_hits_any) if selected_hits_any else 0.0),
        "baseline_first_pct": 100.0 * float(first_hits) / max(1, n),
        "baseline_shortest_pct": 100.0 * float(shortest_hits) / max(1, n),
        "topk_any_correct_pct": [100.0 * _mean(xs) for xs in topk_hits],
        "topk_any_correct_given_any_pct": [100.0 * (_mean(xs) if xs else 0.0) for xs in topk_hits_any],
    }

    print(json.dumps(out, indent=2, sort_keys=True))

    if str(args.output_json).strip():
        with open(str(args.output_json), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
