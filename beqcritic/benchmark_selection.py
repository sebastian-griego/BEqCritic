"""
CLI: benchmark selection strategies on a grouped candidates JSONL file.

The input should include per-candidate correctness labels, e.g. produced by:
  python -m beqcritic.make_grouped_candidates ...
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass

from .modeling import BeqCritic
from .select import score_candidate_matrix, select_from_score_matrix
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

    def add(self, any_correct: bool, selected_correct: bool, component_size: int, cohesion: float | None, centrality: float | None) -> None:
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
        return "\n".join(lines)


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _bootstrap_ci(values: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    if n_boot <= 0:
        raise ValueError("n_boot must be > 0")
    if not values:
        raise ValueError("Cannot bootstrap empty list")

    rnd = random.Random(seed)
    n = len(values)
    stats: list[float] = []
    for _ in range(int(n_boot)):
        s = 0.0
        for _ in range(n):
            s += float(values[rnd.randrange(n)])
        stats.append(s / n)
    stats.sort()
    lo = stats[int(0.025 * (n_boot - 1))]
    hi = stats[int(0.975 * (n_boot - 1))]
    return float(lo), float(hi)


_TOK_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_'.]*|[0-9]+|[^\s]")


def _tokenize(text: str) -> list[str]:
    return _TOK_RE.findall(text)


def _ngrams(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    out: dict[tuple[str, ...], int] = {}
    if n <= 0:
        return out
    for i in range(0, max(0, len(tokens) - n + 1)):
        g = tuple(tokens[i : i + n])
        out[g] = out.get(g, 0) + 1
    return out


def _bleu_score(hyp: list[str], ref: list[str], max_n: int = 4, smooth: float = 1.0) -> float:
    if not hyp and not ref:
        return 1.0
    if not hyp:
        return 0.0

    log_p_sum = 0.0
    for n in range(1, max_n + 1):
        hyp_ngrams = _ngrams(hyp, n)
        ref_ngrams = _ngrams(ref, n)
        overlap = 0
        total = 0
        for g, c in hyp_ngrams.items():
            total += c
            overlap += min(c, ref_ngrams.get(g, 0))
        p_n = (overlap + smooth) / (total + smooth) if total > 0 else 0.0
        log_p_sum += (1.0 / max_n) * math.log(max(1e-12, p_n))

    bp = 1.0
    if len(hyp) < len(ref) and len(hyp) > 0:
        bp = math.exp(1.0 - (len(ref) / len(hyp)))

    return float(bp * math.exp(log_p_sum))


def _sym_bleu(a: str, b: str) -> float:
    ta = _tokenize(normalize_lean_statement(a))
    tb = _tokenize(normalize_lean_statement(b))
    return 0.5 * (_bleu_score(ta, tb) + _bleu_score(tb, ta))


def _bleu_matrix(candidates: list[str]) -> tuple[list[str], list[list[float]]]:
    norm = [normalize_lean_statement(c) for c in candidates]
    n = len(norm)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        mat[i][i] = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            s = _sym_bleu(norm[i], norm[j])
            mat[i][j] = s
            mat[j][i] = s
    return norm, mat


def _load_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--device", type=str, default="", help="e.g. cuda:0, cuda:1, or cpu (default: auto)")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-problems", type=int, default=0, help="Limit number of problems (0 = no limit)")

    p.add_argument("--cluster-mode", type=str, default="components", choices=["components", "support"])
    p.add_argument("--support-frac", type=float, default=0.7, help="Used when --cluster-mode=support")

    p.add_argument("--thresholds", type=str, default="0.5", help="Comma-separated thresholds, e.g. 0.3,0.4,0.5")
    p.add_argument("--tie-breaks", type=str, default="medoid,shortest,first")
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
    p.add_argument("--bootstrap", type=int, default=0, help="If >0, compute 95% CIs with bootstrap resampling")
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--length-key", type=str, default="ref_len_chars", help="Optional per-problem length field to bucket by")
    p.add_argument("--length-buckets", type=str, default="", help="Comma-separated bucket upper bounds (chars)")
    p.add_argument("--report-buckets", action="store_true")
    p.add_argument("--cand-buckets", type=str, default="", help="Comma-separated bucket upper bounds (candidate count)")
    p.add_argument("--report-cand-buckets", action="store_true")
    p.add_argument("--comp-buckets", type=str, default="", help="Comma-separated bucket upper bounds (selected component size)")
    p.add_argument("--report-comp-buckets", action="store_true")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--top-strategies", type=int, default=0, help="If >0, print only the top-N strategies")
    p.add_argument("--sort-by", type=str, default="acc_any", choices=["acc", "acc_any"])
    args = p.parse_args()

    thresholds = [float(x) for x in _parse_csv(args.thresholds)]
    tie_breaks = _parse_csv(args.tie_breaks)
    cluster_ranks = _parse_csv(args.cluster_ranks)

    device = args.device.strip() or None
    critic = BeqCritic(model_name_or_path=args.model, max_length=args.max_length, device=device)

    if args.mutual_ks:
        mutual_ks = [int(x) for x in _parse_csv(args.mutual_ks)]
    else:
        mutual_ks = [int(args.mutual_k)]

    if args.triangle_prune_margins:
        tri_margins = [float(x) for x in _parse_csv(args.triangle_prune_margins)]
    else:
        tri_margins = [float(args.triangle_prune_margin)]

    configs = []
    for thr in thresholds:
        for tb in tie_breaks:
            for cr in cluster_ranks:
                for mk in mutual_ks:
                    for tm in tri_margins:
                        configs.append((thr, tb, cr, mk, tm))

    metrics = {cfg: Metrics() for cfg in configs}
    baseline_first = Metrics()
    baseline_shortest = Metrics()
    baseline_random = Metrics()
    baseline_bleu_medoid = Metrics()

    n_seen = 0
    any_correct_flags: list[int] = []
    lengths: list[int | None] = []
    cand_counts: list[int] = []
    per_strategy_correct: dict[tuple[float, str, str, int, float], list[int]] = {cfg: [] for cfg in configs}
    per_strategy_comp_size: dict[tuple[float, str, str, int, float], list[int]] = {cfg: [] for cfg in configs}
    first_correct: list[int] = []
    shortest_correct: list[int] = []
    random_correct: list[int] = []
    bleu_medoid_correct: list[int] = []

    rnd = random.Random(int(args.random_seed))

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
        rand_idx = rnd.randrange(len(labels01))
        random_is_correct = int(bool(labels01[rand_idx]))

        norm_bleu, bleu_scores = _bleu_matrix(candidates)
        bleu_res = select_from_score_matrix(
            candidates=candidates,
            norm=norm_bleu,
            scores=bleu_scores,
            threshold=0.0,
            tie_break="medoid",
            component_rank="size",
            mutual_top_k=0,
        )
        bleu_is_correct = int(bool(labels01[int(bleu_res.chosen_index)]))

        baseline_first.add(any_correct, bool(first_is_correct), component_size=1, cohesion=None, centrality=None)
        baseline_shortest.add(any_correct, bool(shortest_is_correct), component_size=1, cohesion=None, centrality=None)
        baseline_random.add(any_correct, bool(random_is_correct), component_size=1, cohesion=None, centrality=None)
        baseline_bleu_medoid.add(
            any_correct,
            bool(bleu_is_correct),
            component_size=1,
            cohesion=None,
            centrality=None,
        )

        first_correct.append(first_is_correct)
        shortest_correct.append(shortest_is_correct)
        random_correct.append(random_is_correct)
        bleu_medoid_correct.append(bleu_is_correct)

        norm_scored, scores = score_candidate_matrix(
            candidates=candidates,
            critic=critic,
            batch_size=args.batch_size,
            symmetric=args.symmetric,
        )

        for thr, tb, cr, mk, tm in configs:
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
                support_frac=args.support_frac,
            )
            is_correct = int(bool(labels01[int(res.chosen_index)]))
            per_strategy_correct[(thr, tb, cr, mk, tm)].append(is_correct)
            per_strategy_comp_size[(thr, tb, cr, mk, tm)].append(int(res.component_size))
            metrics[(thr, tb, cr, mk, tm)].add(
                any_correct=any_correct,
                selected_correct=bool(is_correct),
                component_size=int(res.component_size),
                cohesion=res.component_cohesion,
                centrality=res.chosen_centrality,
            )

        n_seen += 1

    def _report_with_ci(name: str, correct_flags: list[int], m: Metrics) -> None:
        print(name)
        print(m.report())
        if args.bootstrap and args.bootstrap > 0:
            n_boot = int(args.bootstrap)
            seed = int(args.bootstrap_seed)

            overall = [float(x) for x in correct_flags]
            lo, hi = _bootstrap_ci(overall, n_boot=n_boot, seed=seed)
            print(f"Selected correct 95% CI: [{100*lo:.1f}%, {100*hi:.1f}%]")

            any_idx = [i for i, a in enumerate(any_correct_flags) if a == 1]
            if any_idx:
                subset = [float(correct_flags[i]) for i in any_idx]
                lo2, hi2 = _bootstrap_ci(subset, n_boot=n_boot, seed=seed + 1)
                print(f"Selected correct | any correct 95% CI: [{100*lo2:.1f}%, {100*hi2:.1f}%]")
        print()

    _report_with_ci("Baseline first", first_correct, baseline_first)
    _report_with_ci("Baseline shortest", shortest_correct, baseline_shortest)
    _report_with_ci("Baseline random", random_correct, baseline_random)
    _report_with_ci("Baseline BLEU-medoid", bleu_medoid_correct, baseline_bleu_medoid)

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
        _report_bucketed("Baseline random", random_correct, lengths, len_bounds, args.length_key)
        _report_bucketed("Baseline BLEU-medoid", bleu_medoid_correct, lengths, len_bounds, args.length_key)

    if args.report_cand_buckets and cand_bounds:
        _report_bucketed("Baseline first", first_correct, cand_counts, cand_bounds, "n_candidates")
        _report_bucketed("Baseline shortest", shortest_correct, cand_counts, cand_bounds, "n_candidates")
        _report_bucketed("Baseline random", random_correct, cand_counts, cand_bounds, "n_candidates")
        _report_bucketed("Baseline BLEU-medoid", bleu_medoid_correct, cand_counts, cand_bounds, "n_candidates")

    if args.report_comp_buckets and comp_bounds:
        ones = [1] * len(first_correct)
        _report_bucketed("Baseline first", first_correct, ones, comp_bounds, "selected_component_size")
        _report_bucketed("Baseline shortest", shortest_correct, ones, comp_bounds, "selected_component_size")
        _report_bucketed("Baseline random", random_correct, ones, comp_bounds, "selected_component_size")
        _report_bucketed("Baseline BLEU-medoid", bleu_medoid_correct, ones, comp_bounds, "selected_component_size")

    if args.top_strategies and int(args.top_strategies) > 0:
        rows = []
        for thr, tb, cr, mk, tm in configs:
            flags = per_strategy_correct[(thr, tb, cr, mk, tm)]
            acc = sum(flags) / max(1, len(flags))
            any_idx = [i for i, a in enumerate(any_correct_flags) if a == 1]
            acc_any = (sum(flags[i] for i in any_idx) / max(1, len(any_idx))) if any_idx else 0.0
            m = metrics[(thr, tb, cr, mk, tm)]
            rows.append((acc_any if args.sort_by == "acc_any" else acc, acc, acc_any, thr, tb, cr, mk, tm, m))
        rows.sort(key=lambda r: (-r[0], -r[1], r[3], r[4], r[5], r[6], r[7]))

        print(f"Top {int(args.top_strategies)} strategies (sorted by {args.sort_by})")
        for rank, (_, acc, acc_any, thr, tb, cr, mk, tm, m) in enumerate(rows[: int(args.top_strategies)], start=1):
            print(
                f"{rank:>2}. acc={100*acc:.1f}% acc|any={100*acc_any:.1f}% "
                f"thr={thr} tie={tb} rank={cr} mk={mk} tri={tm} "
                f"avg_comp={m.sum_component_size/max(1,m.problems):.2f}"
            )
        return

    for thr, tb, cr, mk, tm in configs:
        name = (
            f"Strategy: threshold={thr} tie_break={tb} cluster_rank={cr} "
            f"symmetric={args.symmetric} mutual_k={mk} tri_margin={tm} mode={args.cluster_mode} support={args.support_frac}"
        )
        _report_with_ci(name, per_strategy_correct[(thr, tb, cr, mk, tm)], metrics[(thr, tb, cr, mk, tm)])
        if args.report_buckets and len_bounds:
            _report_bucketed(name, per_strategy_correct[(thr, tb, cr, mk, tm)], lengths, len_bounds, args.length_key)
        if args.report_cand_buckets and cand_bounds:
            _report_bucketed(name, per_strategy_correct[(thr, tb, cr, mk, tm)], cand_counts, cand_bounds, "n_candidates")
        if args.report_comp_buckets and comp_bounds:
            _report_bucketed(
                name,
                per_strategy_correct[(thr, tb, cr, mk, tm)],
                per_strategy_comp_size[(thr, tb, cr, mk, tm)],
                comp_bounds,
                "selected_component_size",
            )


if __name__ == "__main__":
    main()
