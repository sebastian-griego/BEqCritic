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

from .modeling import BeqCritic
from .select import select_by_equivalence_clustering

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", type=str, default="", help="e.g. cuda:0, cuda:1, or cpu (default: auto)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
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
        "--emit-stats",
        action="store_true",
        help="Include graph statistics (edges/components/isolates) in output JSONL.",
    )
    args = p.parse_args()

    device = args.device.strip() or None
    critic = BeqCritic(model_name_or_path=args.model, max_length=args.max_length, device=device)

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            candidates = obj.get("candidates", [])
            res = select_by_equivalence_clustering(
                candidates=candidates,
                critic=critic,
                threshold=args.threshold,
                batch_size=args.batch_size,
                tie_break=args.tie_break,
                cluster_mode=args.cluster_mode,
                support_frac=args.support_frac,
                component_rank=args.cluster_rank,
                symmetric=args.symmetric,
                mutual_top_k=args.mutual_k,
                triangle_prune_margin=args.triangle_prune_margin,
            )
            out = {
                "problem_id": obj.get("problem_id"),
                "chosen": res.chosen_statement,
                "chosen_index": res.chosen_index,
                "component_size": res.component_size,
                "component_indices": res.component_indices,
            }
            if res.component_cohesion is not None:
                out["component_cohesion"] = res.component_cohesion
            if res.chosen_centrality is not None:
                out["chosen_centrality"] = res.chosen_centrality
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
