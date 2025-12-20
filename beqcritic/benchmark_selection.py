"""
CLI: benchmark selection strategies on a grouped candidates JSONL file.

The input should include per-candidate correctness labels, e.g. produced by:
  python -m beqcritic.make_grouped_candidates ...
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from collections import defaultdict

from .bleu import bleu_medoid_index
from .embedder import TextEmbedder
from .modeling import BeqCritic
from .select import similarity_matrix, select_from_score_matrix, global_medoid_index, knn_medoid_index, ensemble_vote
from .textnorm import normalize_lean_statement


@dataclass
class Metrics:
    problems: int = 0
    has_any_correct: int = 0
    selected_correct: int = 0
    selected_correct_given_any: int = 0
    sum_component_size: float = 0.0
    sum_component_cohesion: float = 0.0
    sum_chosen_centrality: float = 0.0
    n_component_stats: int = 0
    fallback_used: int = 0

    def add(
        self,
        any_correct: bool,
        selected_correct: bool,
        component_size: int,
        cohesion: float | None,
        centrality: float | None,
        fallback_used: bool = False,
    ) -> None:
        self.problems += 1
        self.has_any_correct += int(any_correct)
        self.selected_correct += int(selected_correct)
        self.selected_correct_given_any += int(selected_correct and any_correct)
        self.sum_component_size += float(component_size)
        if cohesion is not None:
            self.sum_component_cohesion += float(cohesion)
            self.n_component_stats += 1
        if centrality is not None:
            self.sum_chosen_centrality += float(centrality)
        self.fallback_used += int(bool(fallback_used))

    def report(self) -> str:
        def pct(x: int, d: int) -> float:
            return 100.0 * x / max(1, d)

        lines = [
            f"Problems: {self.problems}",
            f"Has any correct: {self.has_any_correct} ({pct(self.has_any_correct, self.problems):.1f}%)",
            f"Selected correct: {self.selected_correct} ({pct(self.selected_correct, self.problems):.1f}%)",
        ]
        if self.has_any_correct:
            lines.append(
                "Selected correct | any correct: "
                f"{self.selected_correct_given_any} ({pct(self.selected_correct_given_any, self.has_any_correct):.1f}%)"
            )
        if self.problems:
            lines.append(f"Avg component_size: {self.sum_component_size / self.problems:.2f}")
        if self.n_component_stats:
            lines.append(f"Avg component_cohesion: {self.sum_component_cohesion / self.n_component_stats:.3f}")
        if self.problems:
            lines.append(f"Avg chosen_centrality: {self.sum_chosen_centrality / self.problems:.3f}")
        if self.problems and self.fallback_used:
            lines.append(f"Fallback used: {self.fallback_used} ({pct(self.fallback_used, self.problems):.1f}%)")
        return "\n".join(lines)


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _load_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="", help="Required for --similarity critic|hybrid")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--device", type=str, default="", help="e.g. cuda:0, cuda:1, or cpu (default: auto)")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-problems", type=int, default=0, help="Limit number of problems (0 = no limit)")

    p.add_argument("--similarity", type=str, default="critic", choices=["critic", "bleu", "hybrid"])
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

    p.add_argument("--cluster-mode", type=str, default="components", choices=["components", "support"])
    p.add_argument("--support-frac", type=float, default=0.7, help="Used when --cluster-mode=support")
    p.add_argument("--support-fracs", type=str, default="", help="Optional comma-separated support-frac values")

    p.add_argument("--ensemble", action="store_true", help="Ensemble across all provided configs and vote.")
    p.add_argument("--ensemble-vote", type=str, default="weighted", choices=["weighted", "majority"])

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

    p.add_argument("--thresholds", type=str, default="0.5", help="Comma-separated thresholds, e.g. 0.3,0.4,0.5")
    p.add_argument("--tie-breaks", type=str, default="medoid,shortest,first")
    p.add_argument(
        "--medoid-simple-top-k",
        type=int,
        default=0,
        help="When >0 and tie-break=medoid, pick the simplest candidate among the top-k by centrality.",
    )
    p.add_argument(
        "--medoid-simple-max-drop",
        type=float,
        default=-1.0,
        help="If >=0, only consider candidates with centrality >= (best - max_drop) within the top-k set.",
    )
    p.add_argument("--simple-weight-chars", type=float, default=1.0)
    p.add_argument("--simple-weight-binders", type=float, default=0.5)
    p.add_argument("--simple-weight-prop-assumptions", type=float, default=0.25)
    p.add_argument("--simple-chars-scale", type=float, default=100.0)
    p.add_argument(
        "--cluster-ranks",
        type=str,
        default="size_then_cohesion,size",
        help="Comma-separated: size,size_then_cohesion,cohesion,size_times_cohesion",
    )
    p.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Average score(A,B) and score(B,A). Slower but can reduce order bias.",
    )
    p.add_argument("--mutual-k", type=int, default=0)
    p.add_argument("--mutual-ks", type=str, default="", help="Optional comma-separated mutual-k values")
    p.add_argument("--triangle-prune-margin", type=float, default=0.0)
    p.add_argument("--triangle-prune-margins", type=str, default="", help="Optional comma-separated triangle prune margins")
    p.add_argument("--length-key", type=str, default="ref_len_chars", help="Optional per-problem length field to bucket by")
    p.add_argument("--length-buckets", type=str, default="", help="Comma-separated bucket upper bounds (chars)")
    p.add_argument("--report-buckets", action="store_true")
    p.add_argument("--cand-buckets", type=str, default="", help="Comma-separated bucket upper bounds (candidate count)")
    p.add_argument("--report-cand-buckets", action="store_true")
    p.add_argument("--comp-buckets", type=str, default="", help="Comma-separated bucket upper bounds (selected component size)")
    p.add_argument("--report-comp-buckets", action="store_true")
    p.add_argument("--mbr-knn-k", type=int, default=3, help="k for the Baseline MBR-kNN medoid.")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--top-strategies", type=int, default=0, help="If >0, print only the top-N strategies")
    p.add_argument("--sort-by", type=str, default="acc_any", choices=["acc", "acc_any"])
    args = p.parse_args()

    thresholds = [float(x) for x in _parse_csv(args.thresholds)]
    tie_breaks = _parse_csv(args.tie_breaks)
    cluster_ranks = _parse_csv(args.cluster_ranks)

    fallback_desc = ""
    if args.fallback != "none":
        extra = ""
        if args.fallback == "critic_knn_medoid":
            extra = f" knn_k={int(args.fallback_knn_k)}"
        fallback_desc = (
            f" fallback={args.fallback}{extra}"
            f" min_comp={args.fallback_min_component_size} min_coh={args.fallback_min_cohesion}"
        )

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

    sim_desc = f" similarity={args.similarity}"
    if args.similarity in ["critic", "hybrid"] and used_critic_temperature is not None and float(used_critic_temperature) != 1.0:
        sim_desc += f" temp={float(used_critic_temperature)}"
    if args.similarity == "hybrid":
        sim_desc += f" alpha={float(args.hybrid_alpha)}"
    if args.similarity in ["bleu", "hybrid"] and (int(args.bleu_max_n) != 4 or float(args.bleu_smooth) != 1.0):
        sim_desc += f" bleu_n={int(args.bleu_max_n)} bleu_smooth={float(args.bleu_smooth)}"
    if args.similarity in ["critic", "hybrid"] and str(args.critic_pair_mode) != "all":
        sim_desc += f" pairs={str(args.critic_pair_mode)}"
        if str(args.critic_pair_mode) == "knn":
            sim_desc += f" knn_k={int(args.knn_k)}"

    if args.mutual_ks:
        mutual_ks = [int(x) for x in _parse_csv(args.mutual_ks)]
    else:
        mutual_ks = [int(args.mutual_k)]

    if args.triangle_prune_margins:
        tri_margins = [float(x) for x in _parse_csv(args.triangle_prune_margins)]
    else:
        tri_margins = [float(args.triangle_prune_margin)]

    if args.support_fracs:
        support_fracs = [float(x) for x in _parse_csv(args.support_fracs)]
    else:
        support_fracs = [float(args.support_frac)]
    if args.cluster_mode != "support" and args.support_fracs and len(support_fracs) > 1:
        raise ValueError("--support-fracs is only meaningful with --cluster-mode=support")

    configs = []
    for thr in thresholds:
        for tb in tie_breaks:
            for cr in cluster_ranks:
                for mk in mutual_ks:
                    for tm in tri_margins:
                        for sf in support_fracs:
                            configs.append((thr, tb, cr, mk, tm, sf))

    metrics = {cfg: Metrics() for cfg in configs}
    baseline_first = Metrics()
    baseline_shortest = Metrics()
    baseline_majority = Metrics()
    baseline_random = Metrics()
    baseline_bleu_medoid = Metrics()
    baseline_mbr_global = Metrics()
    baseline_mbr_knn = Metrics()
    ensemble_metrics = Metrics()

    n_seen = 0
    any_correct_flags: list[int] = []
    lengths: list[int | None] = []
    cand_counts: list[int] = []
    per_strategy_correct: dict[tuple[float, str, str, int, float, float], list[int]] = {cfg: [] for cfg in configs}
    per_strategy_comp_size: dict[tuple[float, str, str, int, float, float], list[int]] = {cfg: [] for cfg in configs}
    first_correct: list[int] = []
    shortest_correct: list[int] = []
    majority_correct: list[int] = []
    random_correct: list[int] = []
    bleu_medoid_correct: list[int] = []
    mbr_global_correct: list[int] = []
    mbr_knn_correct: list[int] = []
    ensemble_correct: list[int] = []
    ensemble_comp_size: list[int] = []

    if args.ensemble:
        if len(tie_breaks) != 1:
            raise ValueError("--ensemble requires exactly one tie-break (e.g. --tie-breaks medoid)")
        if args.fallback != "none":
            raise ValueError("--ensemble is not supported together with --fallback (run ensemble standalone).")

    rnd = random.Random(int(args.random_seed))
    mbr_knn_k = max(1, int(args.mbr_knn_k))

    for obj in _load_lines(args.input):
        if args.max_problems and n_seen >= int(args.max_problems):
            break
        candidates = obj.get("candidates") or []
        labels = obj.get("labels") or []
        if not candidates:
            continue
        if len(candidates) != len(labels):
            raise ValueError(
                f"Input candidates/labels length mismatch for {obj.get('problem_id')}: "
                f"{len(candidates)} vs {len(labels)}"
            )
        labels01 = [1 if int(x) else 0 for x in labels]
        any_correct = any(labels01)
        any_correct_flags.append(int(any_correct))
        cand_counts.append(int(len(candidates)))
        if args.length_key and args.length_key in obj and obj[args.length_key] is not None:
            try:
                lengths.append(int(obj[args.length_key]))
            except Exception:
                lengths.append(None)
        else:
            lengths.append(None)

        norm = [normalize_lean_statement(c) for c in candidates]
        shortest_idx = min(range(len(norm)), key=lambda i: (len(norm[i]), i))

        first_is_correct = int(bool(labels01[0]))
        shortest_is_correct = int(bool(labels01[shortest_idx]))
        # Majority vote over normalized statements (paper-style baseline).
        groups: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(norm):
            groups[str(s)].append(int(i))
        # Pick the most frequent normalized statement; ties -> shorter statement, then earliest occurrence.
        best_stmt, best_idxs = min(
            groups.items(),
            key=lambda kv: (-len(kv[1]), len(kv[0]), min(kv[1])),
        )
        majority_idx = min(best_idxs, key=lambda i: (len(norm[i]), i))
        majority_is_correct = int(bool(labels01[int(majority_idx)]))
        rand_idx = rnd.randrange(len(labels01))
        random_is_correct = int(bool(labels01[rand_idx]))

        bleu_idx, bleu_cent = bleu_medoid_index(candidates)
        bleu_is_correct = int(bool(labels01[int(bleu_idx)]))

        baseline_first.add(any_correct, bool(first_is_correct), component_size=1, cohesion=None, centrality=None)
        baseline_shortest.add(any_correct, bool(shortest_is_correct), component_size=1, cohesion=None, centrality=None)
        baseline_majority.add(any_correct, bool(majority_is_correct), component_size=1, cohesion=None, centrality=None)
        baseline_random.add(any_correct, bool(random_is_correct), component_size=1, cohesion=None, centrality=None)
        baseline_bleu_medoid.add(
            any_correct,
            bool(bleu_is_correct),
            component_size=1,
            cohesion=None,
            centrality=float(bleu_cent),
        )

        first_correct.append(first_is_correct)
        shortest_correct.append(shortest_is_correct)
        majority_correct.append(majority_is_correct)
        random_correct.append(random_is_correct)
        bleu_medoid_correct.append(bleu_is_correct)

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

        mbr_global_idx, mbr_global_cent = global_medoid_index(norm_scored, scores)
        mbr_knn_idx, mbr_knn_cent = knn_medoid_index(norm_scored, scores, k=int(mbr_knn_k))
        mbr_global_is_correct = int(bool(labels01[int(mbr_global_idx)]))
        mbr_knn_is_correct = int(bool(labels01[int(mbr_knn_idx)]))

        baseline_mbr_global.add(
            any_correct=any_correct,
            selected_correct=bool(mbr_global_is_correct),
            component_size=1,
            cohesion=None,
            centrality=float(mbr_global_cent),
        )
        baseline_mbr_knn.add(
            any_correct=any_correct,
            selected_correct=bool(mbr_knn_is_correct),
            component_size=1,
            cohesion=None,
            centrality=float(mbr_knn_cent),
        )
        mbr_global_correct.append(int(mbr_global_is_correct))
        mbr_knn_correct.append(int(mbr_knn_is_correct))

        per_problem_results = []
        for thr, tb, cr, mk, tm, sf in configs:
            res = select_from_score_matrix(
                candidates=candidates,
                norm=norm_scored,
                scores=scores,
                threshold=thr,
                tie_break=tb,
                component_rank=cr,
                mutual_top_k=mk,
                triangle_prune_margin=tm,
                cluster_mode=args.cluster_mode,
                support_frac=float(sf),
                medoid_simple_top_k=int(args.medoid_simple_top_k),
                medoid_simple_max_drop=float(args.medoid_simple_max_drop),
                simple_weight_chars=float(args.simple_weight_chars),
                simple_weight_binders=float(args.simple_weight_binders),
                simple_weight_prop_assumptions=float(args.simple_weight_prop_assumptions),
                simple_chars_scale=float(args.simple_chars_scale),
            )
            per_problem_results.append(res)
            chosen_index = int(res.chosen_index)
            used_fallback = False
            if args.fallback != "none":
                need = False
                if args.fallback_min_component_size and int(res.component_size) < int(args.fallback_min_component_size):
                    need = True
                if args.fallback_min_cohesion and res.component_cohesion is not None and float(res.component_cohesion) < float(args.fallback_min_cohesion):
                    need = True
                if need:
                    used_fallback = True
                    if args.fallback == "bleu_medoid":
                        chosen_index = int(bleu_idx)
                    elif args.fallback == "critic_medoid":
                        fb_idx, _ = global_medoid_index(norm_scored, scores)
                        chosen_index = int(fb_idx)
                    elif args.fallback == "critic_knn_medoid":
                        fb_idx, _ = knn_medoid_index(norm_scored, scores, k=int(args.fallback_knn_k))
                        chosen_index = int(fb_idx)
                    else:
                        raise ValueError(f"Unknown fallback={args.fallback!r}")

            is_correct = int(bool(labels01[int(chosen_index)]))
            per_strategy_correct[(thr, tb, cr, mk, tm, sf)].append(is_correct)
            per_strategy_comp_size[(thr, tb, cr, mk, tm, sf)].append(int(res.component_size))
            metrics[(thr, tb, cr, mk, tm, sf)].add(
                any_correct=any_correct,
                selected_correct=bool(is_correct),
                component_size=int(res.component_size),
                cohesion=res.component_cohesion,
                centrality=res.chosen_centrality,
                fallback_used=bool(used_fallback),
            )

        if args.ensemble:
            ens = ensemble_vote(per_problem_results, norm_scored, vote=args.ensemble_vote)
            ens_idx = int(ens.chosen_index)
            ens_correct_flag = int(bool(labels01[ens_idx]))
            ensemble_correct.append(ens_correct_flag)
            ensemble_comp_size.append(int(ens.component_size))
            ensemble_metrics.add(
                any_correct=any_correct,
                selected_correct=bool(ens_correct_flag),
                component_size=int(ens.component_size),
                cohesion=ens.component_cohesion,
                centrality=ens.chosen_centrality,
            )

        n_seen += 1

    def _report(name: str, m: Metrics) -> None:
        print(name)
        print(m.report())
        print()

    _report("Baseline first", baseline_first)
    _report("Baseline shortest", baseline_shortest)
    _report("Baseline majority", baseline_majority)
    _report("Baseline random", baseline_random)
    _report("Baseline BLEU-medoid", baseline_bleu_medoid)
    _report(f"Baseline MBR-global{sim_desc}", baseline_mbr_global)
    _report(f"Baseline MBR-kNN(k={mbr_knn_k}){sim_desc}", baseline_mbr_knn)
    if args.ensemble:
        _report(
            f"Ensemble ({args.ensemble_vote} vote over {len(configs)} configs)",
            ensemble_metrics,
        )

    def _bucket_idx(val: int, bounds: list[int]) -> int:
        for i, b in enumerate(bounds):
            if val <= b:
                return i
        return len(bounds)

    def _report_bucketed(
        name: str,
        correct_flags: list[int],
        values: list[int | None],
        bounds: list[int],
        label: str,
    ) -> None:
        if not bounds:
            return
        buckets: dict[int, list[int]] = {}
        buckets_any: dict[int, list[int]] = {}
        for i, v in enumerate(values):
            if v is None:
                continue
            bi = _bucket_idx(int(v), bounds)
            buckets.setdefault(bi, []).append(correct_flags[i])
            if any_correct_flags[i] == 1:
                buckets_any.setdefault(bi, []).append(correct_flags[i])

        print(f"{name} buckets ({label}) bounds={bounds}")
        for bi in range(len(bounds) + 1):
            xs = buckets.get(bi, [])
            if not xs:
                continue
            acc = 100.0 * sum(xs) / len(xs)
            xs_any = buckets_any.get(bi, [])
            if xs_any:
                acc_any = 100.0 * sum(xs_any) / len(xs_any)
                print(f"  bucket {bi}: {len(xs)} problems acc={acc:.1f}% acc|any={acc_any:.1f}%")
            else:
                print(f"  bucket {bi}: {len(xs)} problems acc={acc:.1f}%")
        print()

    len_bounds: list[int] = []
    if args.report_buckets:
        if args.length_buckets:
            len_bounds = sorted({int(x) for x in _parse_csv(args.length_buckets)})
        else:
            lens = sorted([l for l in lengths if l is not None])
            if len(lens) >= 3:
                len_bounds = [lens[len(lens) // 3], lens[(2 * len(lens)) // 3]]

    cand_bounds: list[int] = []
    if args.report_cand_buckets:
        if args.cand_buckets:
            cand_bounds = sorted({int(x) for x in _parse_csv(args.cand_buckets)})
        else:
            cs = sorted(cand_counts)
            if len(cs) >= 3:
                cand_bounds = [cs[len(cs) // 3], cs[(2 * len(cs)) // 3]]

    comp_bounds: list[int] = []
    if args.report_comp_buckets:
        if args.comp_buckets:
            comp_bounds = sorted({int(x) for x in _parse_csv(args.comp_buckets)})
        else:
            comp_bounds = [1, 2, 3, 5, 10]

    if args.report_buckets and len_bounds:
        _report_bucketed("Baseline first", first_correct, lengths, len_bounds, args.length_key)
        _report_bucketed("Baseline shortest", shortest_correct, lengths, len_bounds, args.length_key)
        _report_bucketed("Baseline majority", majority_correct, lengths, len_bounds, args.length_key)
        _report_bucketed("Baseline random", random_correct, lengths, len_bounds, args.length_key)
        _report_bucketed("Baseline BLEU-medoid", bleu_medoid_correct, lengths, len_bounds, args.length_key)
        _report_bucketed(f"Baseline MBR-global{sim_desc}", mbr_global_correct, lengths, len_bounds, args.length_key)
        _report_bucketed(
            f"Baseline MBR-kNN(k={mbr_knn_k}){sim_desc}", mbr_knn_correct, lengths, len_bounds, args.length_key
        )
        if args.ensemble:
            _report_bucketed(
                f"Ensemble ({args.ensemble_vote})",
                ensemble_correct,
                lengths,
                len_bounds,
                args.length_key,
            )

    if args.report_cand_buckets and cand_bounds:
        _report_bucketed("Baseline first", first_correct, cand_counts, cand_bounds, "n_candidates")
        _report_bucketed("Baseline shortest", shortest_correct, cand_counts, cand_bounds, "n_candidates")
        _report_bucketed("Baseline majority", majority_correct, cand_counts, cand_bounds, "n_candidates")
        _report_bucketed("Baseline random", random_correct, cand_counts, cand_bounds, "n_candidates")
        _report_bucketed("Baseline BLEU-medoid", bleu_medoid_correct, cand_counts, cand_bounds, "n_candidates")
        _report_bucketed(f"Baseline MBR-global{sim_desc}", mbr_global_correct, cand_counts, cand_bounds, "n_candidates")
        _report_bucketed(
            f"Baseline MBR-kNN(k={mbr_knn_k}){sim_desc}", mbr_knn_correct, cand_counts, cand_bounds, "n_candidates"
        )
        if args.ensemble:
            _report_bucketed(
                f"Ensemble ({args.ensemble_vote})",
                ensemble_correct,
                cand_counts,
                cand_bounds,
                "n_candidates",
            )

    if args.report_comp_buckets and comp_bounds:
        ones = [1] * len(first_correct)
        _report_bucketed("Baseline first", first_correct, ones, comp_bounds, "selected_component_size")
        _report_bucketed("Baseline shortest", shortest_correct, ones, comp_bounds, "selected_component_size")
        _report_bucketed("Baseline majority", majority_correct, ones, comp_bounds, "selected_component_size")
        _report_bucketed("Baseline random", random_correct, ones, comp_bounds, "selected_component_size")
        _report_bucketed("Baseline BLEU-medoid", bleu_medoid_correct, ones, comp_bounds, "selected_component_size")
        _report_bucketed(f"Baseline MBR-global{sim_desc}", mbr_global_correct, ones, comp_bounds, "selected_component_size")
        _report_bucketed(
            f"Baseline MBR-kNN(k={mbr_knn_k}){sim_desc}", mbr_knn_correct, ones, comp_bounds, "selected_component_size"
        )
        if args.ensemble:
            _report_bucketed(
                f"Ensemble ({args.ensemble_vote})",
                ensemble_correct,
                ensemble_comp_size,
                comp_bounds,
                "selected_component_size",
            )

    if args.top_strategies and int(args.top_strategies) > 0:
        rows = []
        for thr, tb, cr, mk, tm, sf in configs:
            flags = per_strategy_correct[(thr, tb, cr, mk, tm, sf)]
            acc = sum(flags) / max(1, len(flags))
            any_idx = [i for i, a in enumerate(any_correct_flags) if a == 1]
            acc_any = (sum(flags[i] for i in any_idx) / max(1, len(any_idx))) if any_idx else 0.0
            m = metrics[(thr, tb, cr, mk, tm, sf)]
            rows.append((acc_any if args.sort_by == "acc_any" else acc, acc, acc_any, thr, tb, cr, mk, tm, sf, m))
        rows.sort(key=lambda r: (-r[0], -r[1], r[3], r[4], r[5], r[6], r[7], r[8]))

        print(f"Top {int(args.top_strategies)} strategies (sorted by {args.sort_by})")
        print(f"Note:{sim_desc.strip()}")
        if fallback_desc:
            print(f"Note:{fallback_desc}")
        for rank, (_, acc, acc_any, thr, tb, cr, mk, tm, sf, m) in enumerate(
            rows[: int(args.top_strategies)], start=1
        ):
            support_desc = ""
            if args.cluster_mode == "support" and len(support_fracs) > 1:
                support_desc = f" support={sf}"
            print(
                f"{rank:>2}. acc={100*acc:.1f}% acc|any={100*acc_any:.1f}% "
                f"thr={thr} tie={tb} rank={cr} mk={mk} tri={tm} "
                f"avg_comp={m.sum_component_size/max(1,m.problems):.2f}{support_desc}"
            )
        return

    for thr, tb, cr, mk, tm, sf in configs:
        support_desc = ""
        if args.cluster_mode == "support":
            support_desc = f" support={sf}"
        name = (
            f"Strategy:{sim_desc} threshold={thr} tie_break={tb} cluster_rank={cr} "
            f"symmetric={args.symmetric} mutual_k={mk} tri_margin={tm} mode={args.cluster_mode}{support_desc}"
            f"{fallback_desc}"
        )
        _report(name, metrics[(thr, tb, cr, mk, tm, sf)])
        if args.report_buckets and len_bounds:
            _report_bucketed(
                name,
                per_strategy_correct[(thr, tb, cr, mk, tm, sf)],
                lengths,
                len_bounds,
                args.length_key,
            )
        if args.report_cand_buckets and cand_bounds:
            _report_bucketed(
                name,
                per_strategy_correct[(thr, tb, cr, mk, tm, sf)],
                cand_counts,
                cand_bounds,
                "n_candidates",
            )
        if args.report_comp_buckets and comp_bounds:
            _report_bucketed(
                name,
                per_strategy_correct[(thr, tb, cr, mk, tm, sf)],
                per_strategy_comp_size[(thr, tb, cr, mk, tm, sf)],
                comp_bounds,
                "selected_component_size",
            )


if __name__ == "__main__":
    main()
