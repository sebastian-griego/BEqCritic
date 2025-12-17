"""
CLI: benchmark selection strategies on a grouped candidates JSONL file.

The input should include per-candidate correctness labels, e.g. produced by:
  python -m beqcritic.make_grouped_candidates ...
"""
from __future__ import annotations

import argparse
import json
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


def _load_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-problems", type=int, default=0, help="Limit number of problems (0 = no limit)")

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
    args = p.parse_args()

    thresholds = [float(x) for x in _parse_csv(args.thresholds)]
    tie_breaks = _parse_csv(args.tie_breaks)
    cluster_ranks = _parse_csv(args.cluster_ranks)

    critic = BeqCritic(model_name_or_path=args.model, max_length=args.max_length)

    configs = []
    for thr in thresholds:
        for tb in tie_breaks:
            for cr in cluster_ranks:
                configs.append((thr, tb, cr))

    metrics = {cfg: Metrics() for cfg in configs}
    baseline_first = Metrics()
    baseline_shortest = Metrics()

    n_seen = 0
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

        norm = [normalize_lean_statement(c) for c in candidates]
        shortest_idx = min(range(len(norm)), key=lambda i: (len(norm[i]), i))

        baseline_first.add(any_correct, bool(labels01[0]), component_size=1, cohesion=None, centrality=None)
        baseline_shortest.add(any_correct, bool(labels01[shortest_idx]), component_size=1, cohesion=None, centrality=None)

        norm_scored, scores = score_candidate_matrix(
            candidates=candidates,
            critic=critic,
            batch_size=args.batch_size,
            symmetric=args.symmetric,
        )

        for thr, tb, cr in configs:
            res = select_from_score_matrix(
                candidates=candidates,
                norm=norm_scored,
                scores=scores,
                threshold=thr,
                tie_break=tb,
                component_rank=cr,
                mutual_top_k=args.mutual_k,
            )
            metrics[(thr, tb, cr)].add(
                any_correct=any_correct,
                selected_correct=bool(labels01[int(res.chosen_index)]),
                component_size=int(res.component_size),
                cohesion=res.component_cohesion,
                centrality=res.chosen_centrality,
            )

        n_seen += 1

    print("Baseline first")
    print(baseline_first.report())
    print()
    print("Baseline shortest")
    print(baseline_shortest.report())
    print()

    for thr, tb, cr in configs:
        print(f"Strategy: threshold={thr} tie_break={tb} cluster_rank={cr} symmetric={args.symmetric} mutual_k={args.mutual_k}")
        print(metrics[(thr, tb, cr)].report())
        print()


if __name__ == "__main__":
    main()

