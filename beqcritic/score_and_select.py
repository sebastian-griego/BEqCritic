"""
CLI: cluster candidates by learned equivalence and select a representative.

Input JSONL format (one problem per line):
  {"problem_id": "...", "candidates": ["theorem ...", "...", ...]}

Output JSONL format:
  {"problem_id": "...", "chosen": "...", "chosen_index": 3, "component_size": 7, "component_indices": [..]}
"""
from __future__ import annotations

import argparse
import json
import math

from .bleu import bleu_medoid_index, bleu_centrality_ranking
from .embedder import TextEmbedder
from .features import extract_features
from .modeling import BeqCritic
from .select import (
    component_representative_index,
    global_medoid_index,
    global_geometric_medoid_index,
    knn_medoid_index,
    ranked_components_from_scores,
    similarity_matrix,
    select_from_score_matrix,
)

def _centralities_from_scores(
    scores: list[list[float]],
    *,
    objective: str,
    eps: float = 1e-6,
) -> list[float]:
    n = len(scores)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    obj = str(objective).strip().lower()
    if obj not in ["mean", "gmean"]:
        raise ValueError(f"Unknown medoid objective={objective!r}")
    if obj == "mean":
        cents: list[float] = []
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j == i:
                    continue
                s += float(scores[i][j])
            cents.append(s / max(1, n - 1))
        return cents

    e = float(eps)
    if e <= 0:
        raise ValueError(f"eps must be > 0, got {eps!r}")
    cents = []
    for i in range(n):
        s = 0.0
        for j in range(n):
            if j == i:
                continue
            x = float(scores[i][j])
            if x < e:
                x = e
            elif x > 1.0:
                x = 1.0
            s += math.log(x)
        cents.append(math.exp(s / max(1, n - 1)))
    return cents

def _global_cohesion(scores: list[list[float]]) -> float | None:
    n = len(scores)
    if n <= 1:
        return 1.0
    s = 0.0
    m = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += float(scores[i][j])
            m += 1
    return s / max(1, m)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="", help="Required for --similarity critic|hybrid")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", type=str, default="", help="e.g. cuda:0, cuda:1, or cpu (default: auto)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--similarity", type=str, default="critic", choices=["critic", "bleu", "hybrid"])
    p.add_argument(
        "--select-mode",
        type=str,
        default="cluster",
        choices=["cluster", "global_medoid"],
        help="Selection rule: cluster (thresholded graph + component rep) or global_medoid (Self-BLEU-like consensus).",
    )
    p.add_argument(
        "--medoid-objective",
        type=str,
        default="mean",
        choices=["mean", "gmean"],
        help="Objective for --select-mode=global_medoid (mean or geometric-mean centrality).",
    )
    p.add_argument(
        "--critic-pair-mode",
        type=str,
        default="all",
        choices=["all", "knn"],
        help="How to choose which candidate pairs are scored by the critic.",
    )
    p.add_argument("--knn-k", type=int, default=10, help="k for --critic-pair-mode=knn (kNN over embeddings).")
    p.add_argument(
        "--knn-embed-model",
        type=str,
        default="",
        help="Embedding model for --critic-pair-mode=knn (default: same as --model).",
    )
    p.add_argument("--knn-embed-max-length", type=int, default=256)
    p.add_argument("--knn-embed-batch-size", type=int, default=32)
    p.add_argument(
        "--knn-mutual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, keep only mutual kNN edges for --critic-pair-mode=knn.",
    )
    p.add_argument("--knn-chunk-size", type=int, default=1024, help="Chunk size for kNN similarity computation.")
    p.add_argument(
        "--critic-temperature",
        type=float,
        default=None,
        help=(
            "Temperature scaling for critic probabilities (only for --similarity critic|hybrid). "
            "If unset, uses the calibrated value from the checkpoint (temperature.json) when available."
        ),
    )
    p.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.5,
        help="Score = alpha*critic + (1-alpha)*BLEU (only for --similarity hybrid).",
    )
    p.add_argument("--bleu-max-n", type=int, default=4, help="Used for --similarity bleu|hybrid.")
    p.add_argument("--bleu-smooth", type=float, default=1.0, help="Used for --similarity bleu|hybrid.")
    p.add_argument("--tie-break", type=str, default="medoid", choices=["medoid", "shortest", "first"])
    p.add_argument(
        "--medoid-simple-top-k",
        type=int,
        default=0,
        help=(
            "When >0 and medoid selection is used, pick the simplest candidate among the top-k by centrality "
            "(penalizes length/binders/Prop assumptions)."
        ),
    )
    p.add_argument(
        "--medoid-simple-max-drop",
        type=float,
        default=-1.0,
        help="If >=0, only consider candidates with centrality >= (best - max_drop) within the top-k set.",
    )
    p.add_argument("--simple-weight-chars", type=float, default=1.0, help="Penalty weight for chars/--simple-chars-scale.")
    p.add_argument("--simple-weight-binders", type=float, default=0.5, help="Penalty weight for binder count.")
    p.add_argument(
        "--simple-weight-prop-assumptions",
        type=float,
        default=0.25,
        help="Penalty weight for Prop-assumption count (heuristic).",
    )
    p.add_argument(
        "--simple-chars-scale",
        type=float,
        default=100.0,
        help="Scale factor for char length in the simplicity penalty (penalty uses chars/scale).",
    )
    p.add_argument("--cluster-mode", type=str, default="components", choices=["components", "support"])
    p.add_argument("--support-frac", type=float, default=0.7, help="Used when --cluster-mode=support")
    p.add_argument(
        "--cluster-rank",
        type=str,
        default="size_then_cohesion",
        choices=["size", "size_then_cohesion", "cohesion", "size_times_cohesion"],
    )
    p.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Average score(A,B) and score(B,A). Slower but can reduce order bias.",
    )
    p.add_argument(
        "--mutual-k",
        type=int,
        default=0,
        help="If >0, keep edges only when i and j are mutual top-k neighbors (reduces bridge errors).",
    )
    p.add_argument(
        "--triangle-prune-margin",
        type=float,
        default=0.2,
        help="If >0, prune inconsistent triangles where AB and BC are strong but AC is weak.",
    )
    p.add_argument(
        "--fallback",
        type=str,
        default="none",
        choices=["none", "bleu_medoid", "critic_medoid", "critic_knn_medoid"],
    )
    p.add_argument(
        "--fallback-min-component-size",
        type=int,
        default=0,
        help="If >0, use fallback when chosen component_size < this value.",
    )
    p.add_argument(
        "--fallback-min-cohesion",
        type=float,
        default=0.0,
        help="If >0, use fallback when chosen component_cohesion < this value.",
    )
    p.add_argument(
        "--fallback-knn-k",
        type=int,
        default=3,
        help="k for --fallback=critic_knn_medoid (mean of top-k critic similarities).",
    )
    p.add_argument(
        "--emit-stats",
        action="store_true",
        help="Include graph statistics (edges/components/isolates) in output JSONL.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="If >1, also emit a ranked list of top-k candidate indices (cluster representatives).",
    )
    p.add_argument(
        "--emit-topk-text",
        action="store_true",
        help="When used with --top-k > 1, also include the selected top-k Lean strings in the output JSONL.",
    )
    args = p.parse_args()

    critic = None
    used_critic_temperature = None
    knn_embedder = None
    if args.similarity in ["critic", "hybrid"]:
        if not args.model:
            raise ValueError("--model is required when --similarity is critic or hybrid")
        device = args.device.strip() or None
        critic = BeqCritic(model_name_or_path=args.model, max_length=args.max_length, device=device)
        used_critic_temperature = (
            float(args.critic_temperature) if args.critic_temperature is not None else float(critic.temperature)
        )
        if args.critic_pair_mode == "knn":
            embed_model = args.knn_embed_model.strip() or args.model
            knn_embedder = TextEmbedder(
                model_name_or_path=embed_model,
                max_length=int(args.knn_embed_max_length),
                device=str(critic.device) if critic.device is not None else None,
            )

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            candidates = obj.get("candidates", [])
            norm_scored, scores = similarity_matrix(
                candidates=candidates,
                critic=critic,
                batch_size=args.batch_size,
                symmetric=args.symmetric,
                similarity=args.similarity,
                critic_temperature=args.critic_temperature,
                critic_pair_mode=str(args.critic_pair_mode),
                knn_embedder=knn_embedder,
                knn_k=int(args.knn_k),
                knn_mutual=bool(args.knn_mutual),
                knn_chunk_size=int(args.knn_chunk_size),
                knn_embed_batch_size=int(args.knn_embed_batch_size),
                hybrid_alpha=args.hybrid_alpha,
                bleu_max_n=args.bleu_max_n,
                bleu_smooth=args.bleu_smooth,
            )
            if str(args.select_mode) == "cluster":
                res = select_from_score_matrix(
                    candidates=candidates,
                    norm=norm_scored,
                    scores=scores,
                    threshold=args.threshold,
                    tie_break=args.tie_break,
                    component_rank=args.cluster_rank,
                    mutual_top_k=args.mutual_k,
                    triangle_prune_margin=args.triangle_prune_margin,
                    cluster_mode=args.cluster_mode,
                    support_frac=args.support_frac,
                    medoid_simple_top_k=int(args.medoid_simple_top_k),
                    medoid_simple_max_drop=float(args.medoid_simple_max_drop),
                    simple_weight_chars=float(args.simple_weight_chars),
                    simple_weight_binders=float(args.simple_weight_binders),
                    simple_weight_prop_assumptions=float(args.simple_weight_prop_assumptions),
                    simple_chars_scale=float(args.simple_chars_scale),
                )

                primary_chosen_index = int(res.chosen_index)
                primary_chosen_statement = str(res.chosen_statement)
                component_indices = list(res.component_indices)
                component_size = int(res.component_size)
                component_cohesion = res.component_cohesion
                chosen_centrality = res.chosen_centrality
                chosen_centrality_gap = res.chosen_centrality_gap
                stats = res
            else:
                if str(args.similarity) in ["critic", "hybrid"] and str(args.critic_pair_mode) != "all":
                    raise ValueError("--select-mode=global_medoid currently requires --critic-pair-mode=all")

                cents = _centralities_from_scores(scores, objective=str(args.medoid_objective))
                ranked = sorted(
                    range(len(candidates)),
                    key=lambda i: (-float(cents[i]), len(norm_scored[i]), int(i)),
                )
                primary_chosen_index = int(ranked[0]) if ranked else 0
                if int(args.medoid_simple_top_k) > 0 and ranked:
                    k = min(int(args.medoid_simple_top_k), len(ranked))
                    top = ranked[:k]
                    if float(args.medoid_simple_max_drop) >= 0 and top:
                        md = float(args.medoid_simple_max_drop)
                        best = float(cents[int(top[0])])
                        filt = [i for i in top if best - float(cents[int(i)]) <= md]
                        if filt:
                            top = filt
                    if float(args.simple_chars_scale) <= 0:
                        raise ValueError(f"--simple-chars-scale must be > 0, got {args.simple_chars_scale!r}")

                    def _penalty(i: int) -> float:
                        f = extract_features(norm_scored[int(i)])
                        return (
                            float(args.simple_weight_chars) * (float(f.n_chars) / float(args.simple_chars_scale))
                            + float(args.simple_weight_binders) * float(f.n_binders)
                            + float(args.simple_weight_prop_assumptions) * float(f.n_prop_assumptions)
                        )

                    primary_chosen_index = int(
                        min(
                            top,
                            key=lambda i: (
                                float(_penalty(int(i))),
                                -float(cents[int(i)]),
                                len(norm_scored[int(i)]),
                                int(i),
                            ),
                        )
                    )

                chosen_centrality = float(cents[int(primary_chosen_index)]) if cents else None
                primary_chosen_statement = str(candidates[primary_chosen_index])
                component_indices = list(range(len(candidates)))
                component_size = int(len(candidates))
                component_cohesion = _global_cohesion(scores)

                cents_sorted = sorted((float(x) for x in cents), reverse=True)
                if len(cents_sorted) >= 2:
                    chosen_centrality_gap = float(cents_sorted[0]) - float(cents_sorted[1])
                else:
                    chosen_centrality_gap = None
                stats = None

            primary_chosen_index = int(primary_chosen_index)
            chosen_index = int(primary_chosen_index)
            chosen_statement = str(primary_chosen_statement)

            if str(args.select_mode) == "global_medoid":
                suffix = "_gmean" if str(args.medoid_objective) == "gmean" else ""
                selection_method = f"{str(args.similarity)}_medoid{suffix}"
            else:
                selection_method = str(args.similarity)
            fallback_used = False

            if args.fallback != "none":
                need = False
                if args.fallback_min_component_size and component_size < int(args.fallback_min_component_size):
                    need = True
                if args.fallback_min_cohesion and component_cohesion is not None and float(component_cohesion) < float(args.fallback_min_cohesion):
                    need = True
                if need:
                    if args.fallback == "bleu_medoid":
                        fb_idx, fb_cent = bleu_medoid_index(candidates)
                        chosen_index = int(fb_idx)
                        chosen_statement = str(candidates[chosen_index])
                        chosen_centrality = float(fb_cent)
                        chosen_centrality_gap = None
                        selection_method = "bleu_medoid"
                        fallback_used = True
                    elif args.fallback == "critic_medoid":
                        fb_idx, fb_cent = global_medoid_index(norm_scored, scores)
                        chosen_index = int(fb_idx)
                        chosen_statement = str(candidates[chosen_index])
                        chosen_centrality = float(fb_cent)
                        chosen_centrality_gap = None
                        selection_method = "critic_medoid"
                        fallback_used = True
                    elif args.fallback == "critic_knn_medoid":
                        fb_idx, fb_cent = knn_medoid_index(norm_scored, scores, k=int(args.fallback_knn_k))
                        chosen_index = int(fb_idx)
                        chosen_statement = str(candidates[chosen_index])
                        chosen_centrality = float(fb_cent)
                        chosen_centrality_gap = None
                        selection_method = "critic_knn_medoid"
                        fallback_used = True
                    else:
                        raise ValueError(f"Unknown fallback={args.fallback!r}")

            chosen_indices: list[int] | None = None
            if int(args.top_k) > 1:
                k = max(1, int(args.top_k))
                k = min(k, len(candidates))

                if str(args.select_mode) == "global_medoid":
                    cents = _centralities_from_scores(scores, objective=str(args.medoid_objective))
                    ranked = sorted(
                        range(len(candidates)),
                        key=lambda i: (-float(cents[i]), len(norm_scored[i]), i),
                    )
                    chosen_indices = ranked[:k]
                else:
                    comp_stats, _ = ranked_components_from_scores(
                        scores=scores,
                        threshold=float(args.threshold),
                        component_rank=str(args.cluster_rank),
                        mutual_top_k=int(args.mutual_k),
                        triangle_prune_margin=float(args.triangle_prune_margin),
                        triangle_prune_keep_best_edge=True,
                        cluster_mode=str(args.cluster_mode),
                        support_frac=float(args.support_frac),
                    )

                    ranked_reps: list[int] = []
                    for comp, _coh in comp_stats:
                        idx, _cent, _gap = component_representative_index(
                            comp=comp,
                            tie_break=str(args.tie_break),
                            norm=norm_scored,
                            scores=scores,
                            medoid_simple_top_k=int(args.medoid_simple_top_k),
                            medoid_simple_max_drop=float(args.medoid_simple_max_drop),
                            simple_weight_chars=float(args.simple_weight_chars),
                            simple_weight_binders=float(args.simple_weight_binders),
                            simple_weight_prop_assumptions=float(args.simple_weight_prop_assumptions),
                            simple_chars_scale=float(args.simple_chars_scale),
                        )
                        ranked_reps.append(int(idx))

                    fill_order = ranked_reps
                    if fallback_used and args.fallback == "bleu_medoid":
                        fill_order = [i for i, _ in bleu_centrality_ranking(candidates)]

                    out_indices: list[int] = []
                    for idx in [int(chosen_index)] + list(fill_order):
                        if idx < 0 or idx >= len(candidates):
                            continue
                        if idx in out_indices:
                            continue
                        out_indices.append(int(idx))
                        if len(out_indices) >= k:
                            break

                    if len(out_indices) < k:
                        for idx in range(len(candidates)):
                            if idx in out_indices:
                                continue
                            out_indices.append(int(idx))
                            if len(out_indices) >= k:
                                break

                    chosen_indices = out_indices[:k]

            out = {
                "problem_id": obj.get("problem_id"),
                "chosen": chosen_statement,
                "chosen_index": chosen_index,
                "component_size": component_size,
                "component_indices": component_indices,
                "selection_method": selection_method,
                "similarity": str(args.similarity),
                "select_mode": str(args.select_mode),
            }
            if str(args.select_mode) == "global_medoid":
                out["medoid_objective"] = str(args.medoid_objective)
            if int(args.medoid_simple_top_k) > 0:
                out.update(
                    {
                        "medoid_simple_top_k": int(args.medoid_simple_top_k),
                        "medoid_simple_max_drop": float(args.medoid_simple_max_drop),
                        "simple_weight_chars": float(args.simple_weight_chars),
                        "simple_weight_binders": float(args.simple_weight_binders),
                        "simple_weight_prop_assumptions": float(args.simple_weight_prop_assumptions),
                        "simple_chars_scale": float(args.simple_chars_scale),
                    }
                )
            if args.similarity in ["critic", "hybrid"] and used_critic_temperature is not None:
                out["critic_temperature"] = float(used_critic_temperature)
                out["critic_pair_mode"] = str(args.critic_pair_mode)
                if args.critic_pair_mode == "knn":
                    out["knn_k"] = int(args.knn_k)
                    out["knn_mutual"] = bool(args.knn_mutual)
            if args.similarity == "hybrid":
                out["hybrid_alpha"] = float(args.hybrid_alpha)
            if args.similarity in ["bleu", "hybrid"] and (
                int(args.bleu_max_n) != 4 or float(args.bleu_smooth) != 1.0
            ):
                out["bleu_max_n"] = int(args.bleu_max_n)
                out["bleu_smooth"] = float(args.bleu_smooth)
            if component_cohesion is not None:
                out["component_cohesion"] = float(component_cohesion)
            if chosen_centrality is not None:
                out["chosen_centrality"] = float(chosen_centrality)
            if chosen_centrality_gap is not None:
                out["chosen_centrality_gap"] = float(chosen_centrality_gap)
            if chosen_indices is not None:
                out["chosen_indices"] = list(chosen_indices)
                if args.emit_topk_text:
                    out["chosen_topk"] = [str(candidates[int(i)]) for i in chosen_indices]
            if args.fallback != "none":
                out["fallback_used"] = bool(fallback_used)
            if fallback_used:
                out["critic_chosen_index"] = int(primary_chosen_index)
                out["critic_chosen"] = str(primary_chosen_statement)
            if args.emit_stats:
                if stats is not None:
                    out.update(
                        {
                            "edges_before": stats.edges_before,
                            "edges_after": stats.edges_after,
                            "components_before": stats.components_before,
                            "components_after": stats.components_after,
                            "isolated_before": stats.isolated_before,
                            "isolated_after": stats.isolated_after,
                            "edges_readded": stats.edges_readded,
                        }
                    )
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
