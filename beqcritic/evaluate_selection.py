"""
CLI: evaluate selection outputs against grouped candidate labels.

Inputs:
  - candidates JSONL from `beqcritic.make_grouped_candidates` (must include `labels`)
  - selections JSONL from `beqcritic.score_and_select`
"""
from __future__ import annotations

import argparse
import json

from .textnorm import normalize_lean_statement


def _load_jsonl(path: str) -> dict[str, dict]:
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=str, required=True)
    p.add_argument("--selections", type=str, required=True)
    args = p.parse_args()

    cand = _load_jsonl(args.candidates)
    sel = _load_jsonl(args.selections)

    pids = sorted(set(cand.keys()) & set(sel.keys()))
    missing_cand = sorted(set(sel.keys()) - set(cand.keys()))
    missing_sel = sorted(set(cand.keys()) - set(sel.keys()))
    if missing_cand:
        print(f"Warning: {len(missing_cand)} selections missing candidates")
    if missing_sel:
        print(f"Warning: {len(missing_sel)} candidates missing selections")

    n = 0
    n_any = 0
    n_sel_correct = 0
    n_sel_correct_any = 0
    n_first_correct = 0
    n_shortest_correct = 0
    comp_sizes: list[int] = []
    comp_cohesions: list[float] = []
    chosen_centralities: list[float] = []

    for pid in pids:
        c = cand[pid]
        s = sel[pid]

        candidates = c.get("candidates") or []
        labels = c.get("labels") or []
        if len(candidates) != len(labels):
            raise ValueError(f"Candidates/labels length mismatch for {pid}: {len(candidates)} vs {len(labels)}")
        if not candidates:
            continue

        chosen_index = int(s.get("chosen_index"))
        if chosen_index < 0 or chosen_index >= len(labels):
            raise ValueError(f"chosen_index out of range for {pid}: {chosen_index} (n={len(labels)})")

        labels01 = [1 if int(x) else 0 for x in labels]
        any_correct = any(labels01)
        sel_correct = bool(labels01[chosen_index])

        n += 1
        n_any += int(any_correct)
        n_sel_correct += int(sel_correct)
        n_sel_correct_any += int(sel_correct and any_correct)
        n_first_correct += int(bool(labels01[0]))

        norm = [normalize_lean_statement(x) for x in candidates]
        shortest_idx = min(range(len(norm)), key=lambda i: (len(norm[i]), i))
        n_shortest_correct += int(bool(labels01[shortest_idx]))

        if "component_size" in s:
            comp_sizes.append(int(s["component_size"]))
        if "component_cohesion" in s:
            comp_cohesions.append(float(s["component_cohesion"]))
        if "chosen_centrality" in s:
            chosen_centralities.append(float(s["chosen_centrality"]))

    if n == 0:
        raise SystemExit("No overlapping problem_ids to evaluate.")

    def _pct(x: int, d: int) -> float:
        return 100.0 * x / max(1, d)

    print(f"Problems: {n}")
    print(f"Has any correct: {n_any} ({_pct(n_any, n):.1f}%)")
    print(f"Selected correct: {n_sel_correct} ({_pct(n_sel_correct, n):.1f}%)")
    if n_any:
        print(f"Selected correct | any correct: {n_sel_correct_any} ({_pct(n_sel_correct_any, n_any):.1f}%)")
    print(f"Baseline first: {n_first_correct} ({_pct(n_first_correct, n):.1f}%)")
    print(f"Baseline shortest: {n_shortest_correct} ({_pct(n_shortest_correct, n):.1f}%)")
    if comp_sizes:
        avg_comp = sum(comp_sizes) / len(comp_sizes)
        print(f"Avg component_size: {avg_comp:.2f}")
    if comp_cohesions:
        avg_coh = sum(comp_cohesions) / len(comp_cohesions)
        print(f"Avg component_cohesion: {avg_coh:.3f}")
    if chosen_centralities:
        avg_cent = sum(chosen_centralities) / len(chosen_centralities)
        print(f"Avg chosen_centrality: {avg_cent:.3f}")


if __name__ == "__main__":
    main()
