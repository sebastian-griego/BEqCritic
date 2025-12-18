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
    p.add_argument(
        "--max-k",
        type=int,
        default=1,
        help="If selections contain chosen_indices, also report top-k any-correct metrics up to this k.",
    )
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
    max_k = max(1, int(args.max_k))
    topk_correct = [0 for _ in range(max_k)]
    topk_correct_any = [0 for _ in range(max_k)]
    n_first_correct = 0
    n_shortest_correct = 0
    comp_sizes: list[int] = []
    comp_cohesions: list[float] = []
    chosen_centralities: list[float] = []
    edges_before: list[int] = []
    edges_after: list[int] = []
    components_before: list[int] = []
    components_after: list[int] = []
    isolated_before: list[int] = []
    isolated_after: list[int] = []
    edges_readded: list[int] = []

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
                raise ValueError(f"Missing chosen_index in selections for {pid}: {s}")
            chosen_indices = [int(s.get("chosen_index"))]

        bad = [i for i in chosen_indices if i < 0 or i >= len(labels)]
        if bad:
            bad.sort()
            raise ValueError(f"chosen_indices out of range for {pid}: {bad[:5]} (n={len(labels)})")

        chosen_index = int(chosen_indices[0])

        labels01 = [1 if int(x) else 0 for x in labels]
        any_correct = any(labels01)
        sel_correct = bool(labels01[chosen_index])

        n += 1
        n_any += int(any_correct)
        n_sel_correct += int(sel_correct)
        n_sel_correct_any += int(sel_correct and any_correct)

        for k in range(1, max_k + 1):
            picked = chosen_indices[:k]
            hit = any(bool(labels01[i]) for i in picked)
            topk_correct[k - 1] += int(hit)
            topk_correct_any[k - 1] += int(hit and any_correct)
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
        if "edges_before" in s:
            edges_before.append(int(s["edges_before"]))
        if "edges_after" in s:
            edges_after.append(int(s["edges_after"]))
        if "components_before" in s:
            components_before.append(int(s["components_before"]))
        if "components_after" in s:
            components_after.append(int(s["components_after"]))
        if "isolated_before" in s:
            isolated_before.append(int(s["isolated_before"]))
        if "isolated_after" in s:
            isolated_after.append(int(s["isolated_after"]))
        if "edges_readded" in s:
            edges_readded.append(int(s["edges_readded"]))

    if n == 0:
        raise SystemExit("No overlapping problem_ids to evaluate.")

    def _pct(x: int, d: int) -> float:
        return 100.0 * x / max(1, d)

    print(f"Problems: {n}")
    print(f"Has any correct: {n_any} ({_pct(n_any, n):.1f}%)")
    print(f"Selected correct: {n_sel_correct} ({_pct(n_sel_correct, n):.1f}%)")
    if n_any:
        print(f"Selected correct | any correct: {n_sel_correct_any} ({_pct(n_sel_correct_any, n_any):.1f}%)")
    if max_k > 1:
        for k in range(2, max_k + 1):
            print(f"Top-{k} any correct: {topk_correct[k-1]} ({_pct(topk_correct[k-1], n):.1f}%)")
            if n_any:
                print(
                    f"Top-{k} any correct | any correct: {topk_correct_any[k-1]} ({_pct(topk_correct_any[k-1], n_any):.1f}%)"
                )
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
    if edges_before and edges_after:
        print(f"Avg edges_before/after: {sum(edges_before)/len(edges_before):.1f} -> {sum(edges_after)/len(edges_after):.1f}")
    if components_before and components_after:
        print(
            f"Avg components_before/after: {sum(components_before)/len(components_before):.2f} -> "
            f"{sum(components_after)/len(components_after):.2f}"
        )
    if isolated_before and isolated_after:
        print(
            f"Avg isolated_before/after: {sum(isolated_before)/len(isolated_before):.2f} -> "
            f"{sum(isolated_after)/len(isolated_after):.2f}"
        )
    if edges_readded:
        print(f"Avg edges_readded: {sum(edges_readded)/len(edges_readded):.2f}")


if __name__ == "__main__":
    main()
