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

from .bleu import bleu_medoid_index, bleu_centrality_ranking
from .modeling import BeqCritic
from .select import (
    component_representative_index,
    global_medoid_index,
    knn_medoid_index,
    ranked_components_from_scores,
    similarity_matrix,
    select_from_score_matrix,
)

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
        "--critic-temperature",
        type=float,
        default=1.0,
        help="Temperature scaling for critic probabilities (only for --similarity critic|hybrid).",
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
    if args.similarity in ["critic", "hybrid"]:
        if not args.model:
            raise ValueError("--model is required when --similarity is critic or hybrid")
        device = args.device.strip() or None
        critic = BeqCritic(model_name_or_path=args.model, max_length=args.max_length, device=device)

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
                hybrid_alpha=args.hybrid_alpha,
                bleu_max_n=args.bleu_max_n,
                bleu_smooth=args.bleu_smooth,
            )
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
            )

            critic_chosen_index = int(res.chosen_index)
            critic_chosen_statement = str(res.chosen_statement)
            component_indices = list(res.component_indices)
            component_size = int(res.component_size)
            component_cohesion = res.component_cohesion
            chosen_centrality = res.chosen_centrality

            chosen_index = critic_chosen_index
            chosen_statement = critic_chosen_statement
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
                        selection_method = "bleu_medoid"
                        fallback_used = True
                    elif args.fallback == "critic_medoid":
                        fb_idx, fb_cent = global_medoid_index(norm_scored, scores)
                        chosen_index = int(fb_idx)
                        chosen_statement = str(candidates[chosen_index])
                        chosen_centrality = float(fb_cent)
                        selection_method = "critic_medoid"
                        fallback_used = True
                    elif args.fallback == "critic_knn_medoid":
                        fb_idx, fb_cent = knn_medoid_index(norm_scored, scores, k=int(args.fallback_knn_k))
                        chosen_index = int(fb_idx)
                        chosen_statement = str(candidates[chosen_index])
                        chosen_centrality = float(fb_cent)
                        selection_method = "critic_knn_medoid"
                        fallback_used = True
                    else:
                        raise ValueError(f"Unknown fallback={args.fallback!r}")

            chosen_indices: list[int] | None = None
            if int(args.top_k) > 1:
                k = max(1, int(args.top_k))
                k = min(k, len(candidates))

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
                    idx, _cent = component_representative_index(
                        comp=comp,
                        tie_break=str(args.tie_break),
                        norm=norm_scored,
                        scores=scores,
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
            }
            if args.similarity in ["critic", "hybrid"] and float(args.critic_temperature) != 1.0:
                out["critic_temperature"] = float(args.critic_temperature)
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
            if chosen_indices is not None:
                out["chosen_indices"] = list(chosen_indices)
                if args.emit_topk_text:
                    out["chosen_topk"] = [str(candidates[int(i)]) for i in chosen_indices]
            if args.fallback != "none":
                out["fallback_used"] = bool(fallback_used)
            if fallback_used:
                out["critic_chosen_index"] = int(critic_chosen_index)
                out["critic_chosen"] = str(critic_chosen_statement)
            if args.emit_stats:
                out.update(
                    {
                        "edges_before": res.edges_before,
                        "edges_after": res.edges_after,
                        "components_before": res.components_before,
                        "components_after": res.components_after,
                        "isolated_before": res.isolated_before,
                        "isolated_after": res.isolated_after,
                        "edges_readded": res.edges_readded,
                    }
                )
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
