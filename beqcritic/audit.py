from __future__ import annotations

from typing import Any

from .select import SelectionResult, component_representative_index, ranked_components_from_scores


def _shorten(text: str, max_chars: int) -> tuple[str, bool]:
    text = str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[: max(0, max_chars - 3)] + "...", True


def _label_at(labels: list[Any] | None, idx: int) -> int | None:
    if labels is None or idx < 0 or idx >= len(labels):
        return None
    try:
        return int(bool(int(labels[idx])))
    except Exception:
        return None


def _candidate_record(
    idx: int,
    candidates: list[str],
    norm: list[str],
    labels: list[Any] | None,
    max_text_chars: int,
) -> dict[str, Any]:
    text, text_truncated = _shorten(candidates[idx], max_text_chars)
    norm_text, norm_truncated = _shorten(norm[idx], max_text_chars)
    out: dict[str, Any] = {
        "index": int(idx),
        "text": text,
        "text_truncated": bool(text_truncated),
        "normalized": norm_text,
        "normalized_truncated": bool(norm_truncated),
        "normalized_length": int(len(norm[idx])),
    }
    label = _label_at(labels, idx)
    if label is not None:
        out["label"] = int(label)
    return out


def build_selection_audit(
    *,
    problem_id: Any,
    candidates: list[str],
    norm: list[str],
    scores: list[list[float]],
    selected_index: int,
    selection_method: str,
    select_mode: str,
    threshold: float,
    tie_break: str,
    component_rank: str,
    mutual_top_k: int,
    triangle_prune_margin: float,
    triangle_prune_keep_best_edge: bool,
    cluster_mode: str,
    support_frac: float,
    medoid_simple_top_k: int = 0,
    medoid_simple_max_drop: float = -1.0,
    simple_weight_chars: float = 1.0,
    simple_weight_binders: float = 0.5,
    simple_weight_prop_assumptions: float = 0.25,
    simple_chars_scale: float = 100.0,
    selection_result: SelectionResult | None = None,
    labels: list[Any] | None = None,
    fallback_used: bool = False,
    primary_selected_index: int | None = None,
    top_components: int = 5,
    top_edges: int = 30,
    edge_min_score: float = 0.0,
    max_text_chars: int = 500,
) -> dict[str, Any]:
    """
    Build a JSON-serializable explanation for a single selection decision.

    The audit keeps compact candidate text plus top edges/components so large batch
    runs can preserve decision evidence without duplicating the full score matrix.
    """
    n = len(candidates)
    if n == 0:
        raise ValueError("No candidates provided")
    if len(norm) != n:
        raise ValueError("candidates/norm length mismatch")
    if len(scores) != n or any(len(row) != n for row in scores):
        raise ValueError("score matrix shape mismatch")
    selected_index = int(selected_index)
    if selected_index < 0 or selected_index >= n:
        raise ValueError(f"selected_index out of range: {selected_index} (n={n})")

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

    candidates_out = [
        _candidate_record(i, candidates=candidates, norm=norm, labels=labels, max_text_chars=int(max_text_chars))
        for i in range(n)
    ]

    components_out: list[dict[str, Any]] = []
    for rank, (comp, cohesion) in enumerate(comp_stats[: max(0, int(top_components))], start=1):
        rep, rep_cent, rep_gap = component_representative_index(
            comp=list(comp),
            tie_break=str(tie_break),
            norm=norm,
            scores=scores,
            medoid_simple_top_k=int(medoid_simple_top_k),
            medoid_simple_max_drop=float(medoid_simple_max_drop),
            simple_weight_chars=float(simple_weight_chars),
            simple_weight_binders=float(simple_weight_binders),
            simple_weight_prop_assumptions=float(simple_weight_prop_assumptions),
            simple_chars_scale=float(simple_chars_scale),
        )
        item: dict[str, Any] = {
            "rank": int(rank),
            "indices": [int(i) for i in comp],
            "size": int(len(comp)),
            "cohesion": float(cohesion),
            "representative_index": int(rep),
        }
        if rep_cent is not None:
            item["representative_centrality"] = float(rep_cent)
        if rep_gap is not None:
            item["representative_centrality_gap"] = float(rep_gap)
        if labels is not None:
            comp_labels = [_label_at(labels, int(i)) for i in comp]
            comp_labels01 = [int(x) for x in comp_labels if x is not None]
            item["n_labeled_correct"] = int(sum(comp_labels01))
            rep_label = _label_at(labels, int(rep))
            if rep_label is not None:
                item["representative_label"] = int(rep_label)
        components_out.append(item)

    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            score = float(scores[i][j])
            if score >= float(edge_min_score):
                edges.append((score, i, j))
    edges.sort(key=lambda t: (-t[0], t[1], t[2]))

    top_edges_out: list[dict[str, Any]] = []
    for score, i, j in edges[: max(0, int(top_edges))]:
        item = {"i": int(i), "j": int(j), "score": float(score)}
        li = _label_at(labels, i)
        lj = _label_at(labels, j)
        if li is not None:
            item["label_i"] = int(li)
        if lj is not None:
            item["label_j"] = int(lj)
        top_edges_out.append(item)

    chosen_text, chosen_truncated = _shorten(candidates[selected_index], int(max_text_chars))
    selection: dict[str, Any] = {
        "chosen_index": int(selected_index),
        "chosen": chosen_text,
        "chosen_truncated": bool(chosen_truncated),
        "selection_method": str(selection_method),
        "select_mode": str(select_mode),
        "fallback_used": bool(fallback_used),
    }
    chosen_label = _label_at(labels, selected_index)
    if chosen_label is not None:
        selection["chosen_label"] = int(chosen_label)
    if primary_selected_index is not None and int(primary_selected_index) != int(selected_index):
        selection["primary_chosen_index"] = int(primary_selected_index)
        primary_label = _label_at(labels, int(primary_selected_index))
        if primary_label is not None:
            selection["primary_chosen_label"] = int(primary_label)
    if selection_result is not None:
        selection["primary_component_indices"] = [int(i) for i in selection_result.component_indices]
        selection["primary_component_size"] = int(selection_result.component_size)
        if selection_result.component_cohesion is not None:
            selection["primary_component_cohesion"] = float(selection_result.component_cohesion)
        if selection_result.chosen_centrality is not None:
            selection["primary_chosen_centrality"] = float(selection_result.chosen_centrality)
        if selection_result.chosen_centrality_gap is not None:
            selection["primary_chosen_centrality_gap"] = float(selection_result.chosen_centrality_gap)

    return {
        "problem_id": problem_id,
        "n_candidates": int(n),
        "config": {
            "threshold": float(threshold),
            "tie_break": str(tie_break),
            "component_rank": str(component_rank),
            "mutual_top_k": int(mutual_top_k),
            "triangle_prune_margin": float(triangle_prune_margin),
            "triangle_prune_keep_best_edge": bool(triangle_prune_keep_best_edge),
            "cluster_mode": str(cluster_mode),
            "support_frac": float(support_frac),
            "medoid_simple_top_k": int(medoid_simple_top_k),
            "medoid_simple_max_drop": float(medoid_simple_max_drop),
            "simple_weight_chars": float(simple_weight_chars),
            "simple_weight_binders": float(simple_weight_binders),
            "simple_weight_prop_assumptions": float(simple_weight_prop_assumptions),
            "simple_chars_scale": float(simple_chars_scale),
        },
        "selection": selection,
        "graph_stats": {str(k): int(v) for k, v in graph_stats.items()},
        "components": components_out,
        "top_edges": top_edges_out,
        "candidates": candidates_out,
    }
