"""
CLI: debug selection on a single problem from a grouped candidates JSONL.

This is for quick error analysis (bridge edges, cluster splits, BLEU vs critic disagreements).
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from .embedder import TextEmbedder
from .modeling import BeqCritic
from .select import component_representative_index, ranked_components_from_scores, select_from_score_matrix, similarity_matrix
from .textnorm import normalize_lean_statement


def _load_problem(path: str, problem_id: str) -> dict[str, Any]:
    want = str(problem_id)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            pid = obj.get("problem_id")
            if pid is None:
                continue
            if str(pid) == want:
                return obj
    raise SystemExit(f"problem_id={problem_id!r} not found in {path}")


def _shorten(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Grouped candidates JSONL (one problem per line)")
    p.add_argument("--problem-id", type=str, required=True)

    p.add_argument("--model", type=str, default="", help="Required for --similarity critic|hybrid")
    p.add_argument("--device", type=str, default="", help="e.g. cuda:0, cuda:1, or cpu (default: auto)")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)

    p.add_argument("--similarity", type=str, default="critic", choices=["critic", "bleu", "hybrid"])
    p.add_argument("--critic-pair-mode", type=str, default="all", choices=["all", "knn"])
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--knn-embed-model", type=str, default="")
    p.add_argument("--knn-embed-max-length", type=int, default=256)
    p.add_argument("--knn-embed-batch-size", type=int, default=32)
    p.add_argument("--knn-mutual", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--knn-chunk-size", type=int, default=1024)
    p.add_argument("--critic-temperature", type=float, default=None)
    p.add_argument("--hybrid-alpha", type=float, default=0.5)
    p.add_argument("--bleu-max-n", type=int, default=4)
    p.add_argument("--bleu-smooth", type=float, default=1.0)

    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tie-break", type=str, default="medoid", choices=["medoid", "shortest", "first"])
    p.add_argument(
        "--cluster-rank",
        type=str,
        default="size_then_cohesion",
        choices=["size", "size_then_cohesion", "cohesion", "size_times_cohesion"],
    )
    p.add_argument("--cluster-mode", type=str, default="components", choices=["components", "support"])
    p.add_argument("--support-frac", type=float, default=0.7)
    p.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--mutual-k", type=int, default=0)
    p.add_argument("--triangle-prune-margin", type=float, default=0.0)

    p.add_argument("--top-components", type=int, default=5)
    p.add_argument("--top-edges", type=int, default=30)
    p.add_argument("--edge-min-score", type=float, default=0.0, help="Only show edges with score >= this value")
    p.add_argument("--max-text-chars", type=int, default=140)
    args = p.parse_args()

    obj = _load_problem(args.input, args.problem_id)
    candidates = obj.get("candidates") or []
    labels = obj.get("labels", None)
    if not candidates:
        raise SystemExit("No candidates found for this problem")
    if isinstance(labels, list) and len(labels) != len(candidates):
        raise ValueError(f"Candidates/labels length mismatch: {len(candidates)} vs {len(labels)}")

    critic = None
    used_temp = None
    knn_embedder = None
    if args.similarity in ["critic", "hybrid"]:
        if not args.model:
            raise ValueError("--model is required when --similarity is critic or hybrid")
        device = args.device.strip() or None
        critic = BeqCritic(model_name_or_path=args.model, max_length=int(args.max_length), device=device)
        used_temp = float(args.critic_temperature) if args.critic_temperature is not None else float(critic.temperature)
        if args.critic_pair_mode == "knn":
            embed_model = args.knn_embed_model.strip() or args.model
            knn_embedder = TextEmbedder(
                model_name_or_path=embed_model,
                max_length=int(args.knn_embed_max_length),
                device=str(critic.device) if critic.device is not None else None,
            )

    norm, scores = similarity_matrix(
        candidates=candidates,
        similarity=args.similarity,
        critic=critic,
        batch_size=int(args.batch_size),
        symmetric=bool(args.symmetric),
        critic_temperature=args.critic_temperature,
        critic_pair_mode=str(args.critic_pair_mode),
        knn_embedder=knn_embedder,
        knn_k=int(args.knn_k),
        knn_mutual=bool(args.knn_mutual),
        knn_chunk_size=int(args.knn_chunk_size),
        knn_embed_batch_size=int(args.knn_embed_batch_size),
        hybrid_alpha=float(args.hybrid_alpha),
        bleu_max_n=int(args.bleu_max_n),
        bleu_smooth=float(args.bleu_smooth),
    )

    print(f"problem_id: {obj.get('problem_id')}")
    print(f"n_candidates: {len(candidates)}")
    if labels is not None:
        labels01 = [1 if int(x) else 0 for x in labels]
        print(f"n_correct: {sum(labels01)}")
    print(f"similarity: {args.similarity}")
    if used_temp is not None:
        print(f"critic_temperature: {used_temp}")
    if args.similarity == "hybrid":
        print(f"hybrid_alpha: {float(args.hybrid_alpha)}")
    print()

    print("Candidates:")
    for i, (raw, nrm) in enumerate(zip(candidates, norm)):
        lab = ""
        if isinstance(labels, list):
            lab = f" label={int(bool(int(labels[i])))}"
        print(f"  [{i:>2}] len={len(nrm):>4}{lab}  { _shorten(raw, int(args.max_text_chars)) }")
    print()

    res = select_from_score_matrix(
        candidates=candidates,
        norm=norm,
        scores=scores,
        threshold=float(args.threshold),
        tie_break=str(args.tie_break),
        component_rank=str(args.cluster_rank),
        mutual_top_k=int(args.mutual_k),
        triangle_prune_margin=float(args.triangle_prune_margin),
        cluster_mode=str(args.cluster_mode),
        support_frac=float(args.support_frac),
    )

    chosen_idx = int(res.chosen_index)
    chosen_lab = None
    if isinstance(labels, list):
        chosen_lab = int(bool(int(labels[chosen_idx])))

    print("Selection:")
    print(f"  chosen_index: {chosen_idx}")
    if chosen_lab is not None:
        print(f"  chosen_label: {chosen_lab}")
    print(f"  component_size: {int(res.component_size)}")
    if res.component_cohesion is not None:
        print(f"  component_cohesion: {float(res.component_cohesion):.4f}")
    if res.chosen_centrality is not None:
        print(f"  chosen_centrality: {float(res.chosen_centrality):.4f}")
    if res.chosen_centrality_gap is not None:
        print(f"  chosen_centrality_gap: {float(res.chosen_centrality_gap):.4f}")
    print(f"  component_indices: {res.component_indices}")
    print(f"  chosen: {_shorten(candidates[chosen_idx], 400)}")
    print()

    comp_stats, graph_stats = ranked_components_from_scores(
        scores=scores,
        threshold=float(args.threshold),
        component_rank=str(args.cluster_rank),
        mutual_top_k=int(args.mutual_k),
        triangle_prune_margin=float(args.triangle_prune_margin),
        triangle_prune_keep_best_edge=True,
        cluster_mode=str(args.cluster_mode),
        support_frac=float(args.support_frac),
    )

    print("Graph stats:")
    for k in ["edges_before", "edges_after", "components_before", "components_after", "isolated_before", "isolated_after", "edges_readded"]:
        if k in graph_stats:
            print(f"  {k}: {graph_stats[k]}")
    print()

    print(f"Top {int(args.top_components)} components ({args.cluster_rank}):")
    for rank, (comp, coh) in enumerate(comp_stats[: max(0, int(args.top_components))], start=1):
        rep, rep_cent, rep_gap = component_representative_index(
            comp=comp,
            tie_break=str(args.tie_break),
            norm=norm,
            scores=scores,
        )
        extra = ""
        if rep_cent is not None:
            extra += f" rep_cent={rep_cent:.3f}"
        if rep_gap is not None:
            extra += f" rep_gap={rep_gap:.3f}"
        print(f"  {rank:>2}. size={len(comp):>3} coh={coh:.3f} rep={rep}{extra} members={comp[:20]}{'…' if len(comp)>20 else ''}")
    print()

    n = len(candidates)
    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(scores[i][j])
            if s < float(args.edge_min_score):
                continue
            edges.append((s, i, j))
    edges.sort(key=lambda t: (-t[0], t[1], t[2]))

    print(f"Top {int(args.top_edges)} edges (score >= {float(args.edge_min_score)}):")
    for s, i, j in edges[: max(0, int(args.top_edges))]:
        li = ""
        lj = ""
        if isinstance(labels, list):
            li = f" y{i}={int(bool(int(labels[i])))}"
            lj = f" y{j}={int(bool(int(labels[j])))}"
        print(
            f"  {s:.3f}  ({i},{j}){li}{lj}  "
            f"{_shorten(normalize_lean_statement(candidates[i]), 60)}  ||  {_shorten(normalize_lean_statement(candidates[j]), 60)}"
        )


if __name__ == "__main__":
    main()
