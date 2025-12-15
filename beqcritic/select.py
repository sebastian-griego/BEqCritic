"""
Candidate selection via learned equivalence clustering.

Algorithm:
  1) score pairwise equivalence among candidates
  2) add an undirected edge where score >= threshold
  3) take the largest connected component as the consensus class
  4) return a representative candidate

With n=50, the number of unique pairs is 1225, which is practical for a GPU cross-encoder.
"""
from __future__ import annotations

from dataclasses import dataclass

from .textnorm import normalize_lean_statement
from .features import featurize_pair
from .modeling import BeqCritic

@dataclass
class SelectionResult:
    chosen_index: int
    chosen_statement: str
    component_indices: list[int]
    component_size: int

def _with_feats(a: str, b: str) -> tuple[str, str]:
    fa, fb = featurize_pair(a, b)
    return (fa + " " + a, fb + " " + b)

def select_by_equivalence_clustering(
    candidates: list[str],
    critic: BeqCritic,
    threshold: float = 0.5,
    batch_size: int = 16,
    tie_break: str = "shortest",
) -> SelectionResult:
    if not candidates:
        raise ValueError("No candidates provided")

    norm = [normalize_lean_statement(c) for c in candidates]

    idx_pairs = [(i, j) for i in range(len(norm)) for j in range(i + 1, len(norm))]
    pair_texts: list[tuple[str, str]] = []
    for i, j in idx_pairs:
        a, b = _with_feats(norm[i], norm[j])
        pair_texts.append((a, b))

    scores = critic.score_pairs(pair_texts, batch_size=batch_size) if pair_texts else []

    adj = [set([i]) for i in range(len(norm))]
    for (i, j), sc in zip(idx_pairs, scores):
        if sc >= threshold:
            adj[i].add(j)
            adj[j].add(i)

    seen = set()
    comps: list[list[int]] = []
    for i in range(len(norm)):
        if i in seen:
            continue
        stack = [i]
        comp = []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    stack.append(v)
        comps.append(sorted(comp))

    comps.sort(key=lambda c: (-len(c), c[0]))
    best = comps[0]

    if tie_break == "shortest":
        chosen = min(best, key=lambda k: (len(norm[k]), k))
    elif tie_break == "first":
        chosen = best[0]
    else:
        raise ValueError(f"Unknown tie_break={tie_break}")

    return SelectionResult(
        chosen_index=chosen,
        chosen_statement=candidates[chosen],
        component_indices=best,
        component_size=len(best),
    )
