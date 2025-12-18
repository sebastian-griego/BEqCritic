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

@dataclass(frozen=True)
class SelectionConfig:
    threshold: float
    tie_break: str = "medoid"
    component_rank: str = "size_then_cohesion"
    mutual_top_k: int = 0
    triangle_prune_margin: float = 0.0
    triangle_prune_keep_best_edge: bool = True
    cluster_mode: str = "components"
    support_frac: float = 0.7

@dataclass
class SelectionResult:
    chosen_index: int
    chosen_statement: str
    component_indices: list[int]
    component_size: int
    component_cohesion: float | None = None
    chosen_centrality: float | None = None
    edges_before: int | None = None
    edges_after: int | None = None
    components_before: int | None = None
    components_after: int | None = None
    isolated_before: int | None = None
    isolated_after: int | None = None
    edges_readded: int | None = None

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


def _edge_count(adj: list[set[int]]) -> int:
    # Each adjacency set includes the node itself, so subtract self-loops.
    return sum(max(0, len(s) - 1) for s in adj) // 2


def _isolated_count(adj: list[set[int]]) -> int:
    return sum(1 for s in adj if len(s) <= 1)


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

def global_medoid_index(norm: list[str], scores: list[list[float]]) -> tuple[int, float]:
    """
    Return the index of the global medoid under the score matrix.

    The medoid is the candidate with the highest mean score to all other candidates
    (ties broken by shorter normalized statement, then lower index).
    """
    n = len(norm)
    if n == 0:
        raise ValueError("No candidates")
    comp = list(range(n))
    return _medoid(comp, scores, norm)

def knn_medoid_index(norm: list[str], scores: list[list[float]], k: int = 3) -> tuple[int, float]:
    """
    Return the index of the kNN-medoid under the score matrix.

    This picks the candidate with the highest mean similarity to its top-k nearest
    neighbors (ties broken by shorter normalized statement, then lower index).

    Unlike the global medoid, this can be less sensitive to outliers / hubs when a
    candidate set contains multiple weakly-related clusters.
    """
    n = len(norm)
    if n == 0:
        raise ValueError("No candidates")
    if n == 1:
        return 0, 1.0

    kk = int(k)
    kk = max(1, kk)
    kk = min(kk, n - 1)

    best_idx = 0
    best_score = float("-inf")
    for i in range(n):
        vals = [scores[i][j] for j in range(n) if j != i]
        vals.sort(reverse=True)
        c = _mean(vals[:kk])
        if c > best_score:
            best_score = c
            best_idx = i
        elif c == best_score:
            if len(norm[i]) < len(norm[best_idx]):
                best_idx = i
            elif len(norm[i]) == len(norm[best_idx]) and i < best_idx:
                best_idx = i
    return best_idx, float(best_score)

def ensemble_vote(
    results: list[SelectionResult],
    norm: list[str],
    vote: str = "weighted",
) -> SelectionResult:
    """
    Combine multiple SelectionResults into one choice.

    vote:
      - "majority": unweighted vote count
      - "weighted": vote weight = component_size * component_cohesion * chosen_centrality (when available)
    """
    if not results:
        raise ValueError("No results to vote over")
    if vote not in ["majority", "weighted"]:
        raise ValueError(f"Unknown vote={vote!r}")

    vote_counts: dict[int, int] = {}
    vote_weights: dict[int, float] = {}
    per_result_weight: list[float] = []

    for r in results:
        idx = int(r.chosen_index)
        vote_counts[idx] = vote_counts.get(idx, 0) + 1
        w = 1.0
        if vote == "weighted":
            w = float(max(1, int(r.component_size)))
            if r.component_cohesion is not None:
                w *= float(r.component_cohesion)
            if r.chosen_centrality is not None:
                w *= float(r.chosen_centrality)
        per_result_weight.append(float(w))
        vote_weights[idx] = vote_weights.get(idx, 0.0) + float(w)

    best_idx = None
    best_key = None
    for idx in sorted(vote_counts.keys()):
        key = (
            float(vote_weights.get(idx, 0.0)),
            int(vote_counts.get(idx, 0)),
            -len(norm[idx]),
            -idx,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_idx = idx

    assert best_idx is not None

    # Return the "best" member result among those that voted for the winning index.
    best_res = None
    best_res_key = None
    for r, w in zip(results, per_result_weight):
        if int(r.chosen_index) != int(best_idx):
            continue
        key = (
            float(w),
            float(r.component_cohesion) if r.component_cohesion is not None else 0.0,
            float(r.chosen_centrality) if r.chosen_centrality is not None else 0.0,
            -len(norm[int(best_idx)]),
        )
        if best_res_key is None or key > best_res_key:
            best_res_key = key
            best_res = r

    if best_res is None:
        # Fallback (shouldn't happen): pick any result.
        best_res = results[0]
    return best_res


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


def _keep_best_edge_per_isolate(
    adj: list[set[int]],
    scores: list[list[float]],
    threshold: float,
    topk: list[set[int]] | None,
) -> int:
    """
    Guardrail: if pruning isolates a node but it has at least one eligible edge, keep its strongest edge.

    Returns the number of edges re-added.
    """
    n = len(adj)
    readded = 0
    for i in range(n):
        if len(adj[i]) > 1:
            continue

        best_j = None
        best_s = float("-inf")
        for j in range(n):
            if j == i:
                continue
            if scores[i][j] < threshold:
                continue
            if topk is not None:
                if j not in topk[i] or i not in topk[j]:
                    continue
            if scores[i][j] > best_s:
                best_s = scores[i][j]
                best_j = j

        if best_j is None:
            continue
        j = int(best_j)
        if j not in adj[i]:
            adj[i].add(j)
            adj[j].add(i)
            readded += 1
    return readded


def _densify_cluster(
    cluster: set[int],
    adj: list[set[int]],
    support_frac: float,
) -> set[int]:
    """
    Post-process a cluster so every node has >= support_frac connectivity within the cluster.
    """
    if support_frac <= 0:
        return cluster

    changed = True
    while changed and len(cluster) >= 2:
        changed = False
        n = len(cluster)
        min_support = int((support_frac * n) + 0.999999)  # ceil
        # Support counts include self due to adj[u] containing u.
        bad = []
        for u in cluster:
            support = sum(1 for v in cluster if v in adj[u])
            if support < min_support:
                bad.append((support, u))
        if bad:
            bad.sort(key=lambda x: (x[0], x[1]))
            _, u = bad[0]
            cluster.remove(u)
            changed = True
    return cluster


def _support_cluster_from_seed(
    seed: int,
    adj: list[set[int]],
    scores: list[list[float]],
    support_frac: float,
) -> list[int]:
    cluster: set[int] = {int(seed)}
    remaining = set(range(len(adj))) - cluster
    while remaining:
        n = len(cluster)
        min_support = int((support_frac * n) + 0.999999)  # ceil
        best = None
        best_key = None
        for k in remaining:
            support = sum(1 for m in cluster if k in adj[m])
            if support < min_support:
                continue
            avg = _mean([scores[k][m] for m in cluster])
            key = (avg, support, -k)
            if best_key is None or key > best_key:
                best_key = key
                best = k
        if best is None:
            break
        cluster.add(int(best))
        remaining.remove(int(best))

    cluster = _densify_cluster(cluster, adj=adj, support_frac=support_frac)
    return sorted(cluster)


def component_representative_index(
    comp: list[int],
    tie_break: str,
    norm: list[str],
    scores: list[list[float]],
) -> tuple[int, float | None]:
    """
    Pick a representative candidate index from a component.

    Returns (index, centrality) where centrality is only defined for tie_break="medoid".
    """
    if not comp:
        raise ValueError("Empty component")

    if tie_break == "shortest":
        chosen = min(comp, key=lambda k: (len(norm[k]), k))
        return int(chosen), None
    if tie_break == "first":
        return int(comp[0]), None
    if tie_break == "medoid":
        chosen, chosen_cent = _medoid(comp, scores, norm)
        return int(chosen), float(chosen_cent)
    raise ValueError(f"Unknown tie_break={tie_break!r}")


def ranked_components_from_scores(
    scores: list[list[float]],
    threshold: float,
    component_rank: str = "size_then_cohesion",
    mutual_top_k: int = 0,
    triangle_prune_margin: float = 0.0,
    triangle_prune_keep_best_edge: bool = True,
    cluster_mode: str = "components",
    support_frac: float = 0.7,
) -> tuple[list[tuple[list[int], float]], dict[str, int]]:
    """
    Build a thresholded graph and return ranked components.

    Returns:
      - comp_stats: list of (component_indices, component_cohesion), sorted best-first
      - graph_stats: edges/components/isolates counters before/after pruning
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Empty score matrix")

    if any(len(r) != n for r in scores):
        raise ValueError("Internal error: score matrix shape mismatch")

    adj = [set([i]) for i in range(n)]

    topk: list[set[int]] | None = None
    if mutual_top_k and mutual_top_k > 0 and n >= 2:
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

    edges_before = _edge_count(adj)
    comps_before = len(_connected_components(adj))
    isolated_before = _isolated_count(adj)

    _triangle_prune(adj=adj, scores=scores, threshold=threshold, margin=float(triangle_prune_margin))
    edges_readded = 0
    if triangle_prune_keep_best_edge:
        edges_readded = _keep_best_edge_per_isolate(adj=adj, scores=scores, threshold=threshold, topk=topk)

    edges_after = _edge_count(adj)
    comps_after = len(_connected_components(adj))
    isolated_after = _isolated_count(adj)

    if cluster_mode == "components":
        comps = _connected_components(adj)
        comp_stats = []
        for comp in comps:
            coh = _component_cohesion(comp, scores)
            comp_stats.append((comp, coh))
    elif cluster_mode == "support":
        comp_stats = []
        seen = set()
        for seed in range(n):
            comp = _support_cluster_from_seed(seed, adj=adj, scores=scores, support_frac=float(support_frac))
            t = tuple(comp)
            if t in seen:
                continue
            seen.add(t)
            coh = _component_cohesion(comp, scores)
            comp_stats.append((comp, coh))
        if not comp_stats:
            comp_stats = [([0], 1.0)]
    else:
        raise ValueError(f"Unknown cluster_mode={cluster_mode!r}")

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

    graph_stats = {
        "edges_before": int(edges_before),
        "edges_after": int(edges_after),
        "components_before": int(comps_before),
        "components_after": int(comps_after),
        "isolated_before": int(isolated_before),
        "isolated_after": int(isolated_after),
        "edges_readded": int(edges_readded),
    }
    return comp_stats, graph_stats


def select_from_score_matrix(
    candidates: list[str],
    norm: list[str],
    scores: list[list[float]],
    threshold: float = 0.5,
    tie_break: str = "medoid",
    component_rank: str = "size_then_cohesion",
    mutual_top_k: int = 0,
    triangle_prune_margin: float = 0.0,
    triangle_prune_keep_best_edge: bool = True,
    cluster_mode: str = "components",
    support_frac: float = 0.7,
) -> SelectionResult:
    if not candidates:
        raise ValueError("No candidates provided")
    if len(candidates) != len(norm):
        raise ValueError("Internal error: candidates/norm length mismatch")
    if len(scores) != len(candidates) or any(len(r) != len(candidates) for r in scores):
        raise ValueError("Internal error: score matrix shape mismatch")

    comp_stats, graph_stats = ranked_components_from_scores(
        scores=scores,
        threshold=float(threshold),
        component_rank=str(component_rank),
        mutual_top_k=int(mutual_top_k),
        triangle_prune_margin=float(triangle_prune_margin),
        triangle_prune_keep_best_edge=bool(triangle_prune_keep_best_edge),
        cluster_mode=str(cluster_mode),
        support_frac=float(support_frac),
    )
    best_comp, best_coh = comp_stats[0]

    chosen, chosen_cent = component_representative_index(
        comp=best_comp,
        tie_break=str(tie_break),
        norm=norm,
        scores=scores,
    )

    return SelectionResult(
        chosen_index=int(chosen),
        chosen_statement=candidates[int(chosen)],
        component_indices=list(best_comp),
        component_size=len(best_comp),
        component_cohesion=float(best_coh),
        chosen_centrality=float(chosen_cent) if chosen_cent is not None else None,
        edges_before=int(graph_stats["edges_before"]),
        edges_after=int(graph_stats["edges_after"]),
        components_before=int(graph_stats["components_before"]),
        components_after=int(graph_stats["components_after"]),
        isolated_before=int(graph_stats["isolated_before"]),
        isolated_after=int(graph_stats["isolated_after"]),
        edges_readded=int(graph_stats["edges_readded"]),
    )

def select_ensemble_from_score_matrix(
    candidates: list[str],
    norm: list[str],
    scores: list[list[float]],
    configs: list[SelectionConfig],
    vote: str = "weighted",
) -> SelectionResult:
    if not configs:
        raise ValueError("No ensemble configs provided")
    results: list[SelectionResult] = []
    for cfg in configs:
        results.append(
            select_from_score_matrix(
                candidates=candidates,
                norm=norm,
                scores=scores,
                threshold=float(cfg.threshold),
                tie_break=str(cfg.tie_break),
                component_rank=str(cfg.component_rank),
                mutual_top_k=int(cfg.mutual_top_k),
                triangle_prune_margin=float(cfg.triangle_prune_margin),
                triangle_prune_keep_best_edge=bool(cfg.triangle_prune_keep_best_edge),
                cluster_mode=str(cfg.cluster_mode),
                support_frac=float(cfg.support_frac),
            )
        )
    return ensemble_vote(results=results, norm=norm, vote=vote)

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
    triangle_prune_keep_best_edge: bool = True,
    cluster_mode: str = "components",
    support_frac: float = 0.7,
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
        triangle_prune_keep_best_edge=triangle_prune_keep_best_edge,
        cluster_mode=cluster_mode,
        support_frac=support_frac,
    )
