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
from .features import extract_features
from .modeling import BeqCritic

@dataclass
class SelectionResult:
    chosen_index: int
    chosen_statement: str
    component_indices: list[int]
    component_size: int
    component_cohesion: float | None = None
    chosen_centrality: float | None = None

def _mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _score_matrix(
    candidates: list[str],
    critic: BeqCritic,
    batch_size: int,
    symmetric: bool,
) -> tuple[list[str], list[list[float]]]:
    norm = [normalize_lean_statement(c) for c in candidates]
    prefixes = [extract_features(s).to_prefix() for s in norm]
    prefixed = [f"{p} {s}" for p, s in zip(prefixes, norm)]

    idx_pairs = [(i, j) for i in range(len(norm)) for j in range(i + 1, len(norm))]
    pair_texts = [(prefixed[i], prefixed[j]) for i, j in idx_pairs]
    if symmetric and pair_texts:
        pair_texts = pair_texts + [(prefixed[j], prefixed[i]) for i, j in idx_pairs]
        scores_all = critic.score_pairs(pair_texts, batch_size=batch_size)
        half = len(idx_pairs)
        scores_fwd = scores_all[:half]
        scores_rev = scores_all[half:]
        scores = [(a + b) / 2.0 for a, b in zip(scores_fwd, scores_rev)]
    else:
        scores = critic.score_pairs(pair_texts, batch_size=batch_size) if pair_texts else []

    n = len(norm)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        mat[i][i] = 1.0
    for (i, j), sc in zip(idx_pairs, scores):
        s = float(sc)
        mat[i][j] = s
        mat[j][i] = s
    return norm, mat


def score_candidate_matrix(
    candidates: list[str],
    critic: BeqCritic,
    batch_size: int = 16,
    symmetric: bool = False,
) -> tuple[list[str], list[list[float]]]:
    """
    Compute a symmetric NxN score matrix for candidates, where score[i][j] ~= P(equivalent).
    """
    return _score_matrix(candidates=candidates, critic=critic, batch_size=batch_size, symmetric=bool(symmetric))


def _connected_components(adj: list[set[int]]) -> list[list[int]]:
    seen: set[int] = set()
    comps: list[list[int]] = []
    for i in range(len(adj)):
        if i in seen:
            continue
        stack = [i]
        comp: list[int] = []
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
    return comps


def _component_cohesion(comp: list[int], scores: list[list[float]]) -> float:
    if len(comp) <= 1:
        return 1.0
    vals: list[float] = []
    for i in range(len(comp)):
        for j in range(i + 1, len(comp)):
            a = comp[i]
            b = comp[j]
            vals.append(scores[a][b])
    return _mean(vals)


def _medoid(comp: list[int], scores: list[list[float]], norm: list[str]) -> tuple[int, float]:
    if len(comp) == 1:
        return comp[0], 1.0

    best_idx = comp[0]
    best_score = float("-inf")
    for i in comp:
        vals = [scores[i][j] for j in comp if j != i]
        c = _mean(vals)
        if c > best_score:
            best_score = c
            best_idx = i
        elif c == best_score:
            if len(norm[i]) < len(norm[best_idx]):
                best_idx = i
            elif len(norm[i]) == len(norm[best_idx]) and i < best_idx:
                best_idx = i
    return best_idx, float(best_score)


def _triangle_prune(
    adj: list[set[int]],
    scores: list[list[float]],
    threshold: float,
    margin: float,
) -> None:
    if margin <= 0:
        return
    n = len(adj)
    to_remove: set[tuple[int, int]] = set()

    for j in range(n):
        neigh = [i for i in adj[j] if i != j]
        for a in range(len(neigh)):
            i = neigh[a]
            for b in range(a + 1, len(neigh)):
                k = neigh[b]
                s_ij = scores[i][j]
                s_jk = scores[j][k]
                s_ik = scores[i][k]
                if s_ik >= threshold:
                    continue
                if min(s_ij, s_jk) - s_ik < margin:
                    continue
                if s_ij <= s_jk:
                    u, v = (i, j) if i < j else (j, i)
                else:
                    u, v = (j, k) if j < k else (k, j)
                to_remove.add((u, v))

    for u, v in to_remove:
        adj[u].discard(v)
        adj[v].discard(u)


def select_from_score_matrix(
    candidates: list[str],
    norm: list[str],
    scores: list[list[float]],
    threshold: float = 0.5,
    tie_break: str = "medoid",
    component_rank: str = "size_then_cohesion",
    mutual_top_k: int = 0,
    triangle_prune_margin: float = 0.0,
) -> SelectionResult:
    if not candidates:
        raise ValueError("No candidates provided")
    if len(candidates) != len(norm):
        raise ValueError("Internal error: candidates/norm length mismatch")
    if len(scores) != len(candidates) or any(len(r) != len(candidates) for r in scores):
        raise ValueError("Internal error: score matrix shape mismatch")

    n = len(candidates)
    adj = [set([i]) for i in range(n)]

    topk: list[set[int]] | None = None
    if mutual_top_k and mutual_top_k > 0:
        k = min(int(mutual_top_k), n - 1)
        topk = []
        for i in range(n):
            neigh = [(scores[i][j], j) for j in range(n) if j != i]
            neigh.sort(key=lambda x: (-x[0], x[1]))
            topk.append(set([j for _, j in neigh[:k]]))

    for i in range(n):
        for j in range(i + 1, n):
            sc = scores[i][j]
            if sc < threshold:
                continue
            if topk is not None:
                if j not in topk[i] or i not in topk[j]:
                    continue
            adj[i].add(j)
            adj[j].add(i)

    _triangle_prune(adj=adj, scores=scores, threshold=threshold, margin=float(triangle_prune_margin))

    comps = _connected_components(adj)
    comp_stats = []
    for comp in comps:
        coh = _component_cohesion(comp, scores)
        comp_stats.append((comp, coh))

    def _rank_key(comp: list[int], coh: float):
        if component_rank == "size":
            return (-len(comp), comp[0])
        if component_rank == "cohesion":
            return (-coh, -len(comp), comp[0])
        if component_rank == "size_times_cohesion":
            return (-(len(comp) * coh), -len(comp), comp[0])
        if component_rank == "size_then_cohesion":
            return (-len(comp), -coh, comp[0])
        raise ValueError(f"Unknown component_rank={component_rank!r}")

    comp_stats.sort(key=lambda x: _rank_key(x[0], x[1]))
    best_comp, best_coh = comp_stats[0]

    if tie_break == "shortest":
        chosen = min(best_comp, key=lambda k: (len(norm[k]), k))
        chosen_cent = None
    elif tie_break == "first":
        chosen = best_comp[0]
        chosen_cent = None
    elif tie_break == "medoid":
        chosen, chosen_cent = _medoid(best_comp, scores, norm)
    else:
        raise ValueError(f"Unknown tie_break={tie_break!r}")

    return SelectionResult(
        chosen_index=chosen,
        chosen_statement=candidates[chosen],
        component_indices=best_comp,
        component_size=len(best_comp),
        component_cohesion=float(best_coh),
        chosen_centrality=float(chosen_cent) if chosen_cent is not None else None,
    )

def select_by_equivalence_clustering(
    candidates: list[str],
    critic: BeqCritic,
    threshold: float = 0.5,
    batch_size: int = 16,
    tie_break: str = "medoid",
    component_rank: str = "size_then_cohesion",
    symmetric: bool = False,
    mutual_top_k: int = 0,
    triangle_prune_margin: float = 0.0,
) -> SelectionResult:
    norm, scores = _score_matrix(candidates, critic=critic, batch_size=batch_size, symmetric=bool(symmetric))
    return select_from_score_matrix(
        candidates=candidates,
        norm=norm,
        scores=scores,
        threshold=threshold,
        tie_break=tie_break,
        component_rank=component_rank,
        mutual_top_k=mutual_top_k,
        triangle_prune_margin=triangle_prune_margin,
    )
