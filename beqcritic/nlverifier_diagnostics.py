"""Diagnostics for NLVerifier score-based selection outputs.

The CLI consumes grouped candidates with labels plus a selection JSONL that
contains per-candidate scores, such as ``beqcritic.verifier_select
--emit-scores``. It reports selection accuracy, score-ranking quality, and
the missed reachable examples that are most useful for error analysis.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import isfinite, log2
from pathlib import Path
from typing import Any, Iterable

from .jsonl import load_jsonl_map_by_problem_id
from .schema import validate_grouped_candidates
from .statistics import proportion_summary


@dataclass(frozen=True)
class FailureCase:
    problem_id: str
    chosen_index: int
    chosen_score: float
    best_correct_index: int
    best_correct_rank: int
    best_correct_score: float
    score_gap: float
    n_candidates: int
    top_indices: list[int]
    top_scores: list[float]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "chosen_index": self.chosen_index,
            "chosen_score": self.chosen_score,
            "best_correct_index": self.best_correct_index,
            "best_correct_rank": self.best_correct_rank,
            "best_correct_score": self.best_correct_score,
            "score_gap": self.score_gap,
            "n_candidates": self.n_candidates,
            "top_indices": self.top_indices,
            "top_scores": self.top_scores,
        }


def load_jsonl_map(path: str | Path) -> dict[str, dict[str, Any]]:
    return load_jsonl_map_by_problem_id(path, encoding="utf-8-sig")


def analyze_scores(
    *,
    candidates_path: str | Path,
    selections_path: str | Path,
    score_key: str = "scores",
    chosen_index_key: str = "chosen_index",
    top_ks: Iterable[int] = (1, 2, 5, 10),
    minimize: bool = False,
    failure_top_k: int = 5,
) -> dict[str, Any]:
    """Build a JSON-serializable NLVerifier score diagnostics summary."""

    candidate_rows = load_jsonl_map(candidates_path)
    selection_rows = load_jsonl_map(selections_path)
    problem_ids = sorted(set(candidate_rows) & set(selection_rows))
    if not problem_ids:
        raise ValueError("no overlapping problem_ids across candidates and selections")

    top_ks_clean = sorted({max(1, int(k)) for k in top_ks})
    selected_success: list[bool] = []
    any_correct: list[bool] = []
    reciprocal_ranks: list[float] = []
    ndcg_by_k: dict[int, list[float]] = {k: [] for k in top_ks_clean}
    hit_by_k: dict[int, list[bool]] = {k: [] for k in top_ks_clean}
    pooled_labels: list[int] = []
    pooled_scores: list[float] = []
    within_auc_values: list[float] = []
    failures: list[FailureCase] = []
    total_candidates = 0
    skipped_empty = 0
    skipped_missing_scores = 0

    for problem_id in problem_ids:
        grouped = validate_grouped_candidates(
            candidate_rows[problem_id], require_labels=True
        )
        assert grouped.labels is not None
        labels = [1 if int(value) else 0 for value in grouped.labels]
        n = len(grouped.candidates)
        if n == 0:
            skipped_empty += 1
            continue

        selection = selection_rows[problem_id]
        raw_scores = selection.get(score_key)
        if raw_scores is None:
            skipped_missing_scores += 1
            continue
        if not isinstance(raw_scores, list):
            raise ValueError(f"{score_key!r} must be a list for problem_id={problem_id!r}")
        if len(raw_scores) != n:
            raise ValueError(
                f"score length mismatch for {problem_id}: {len(raw_scores)} vs {n}"
            )
        scores = [_as_finite_float(value, problem_id=problem_id) for value in raw_scores]

        chosen_index = _chosen_index(selection, chosen_index_key=chosen_index_key)
        if chosen_index < 0 or chosen_index >= n:
            raise ValueError(
                f"chosen index out of range for {problem_id}: {chosen_index} (n={n})"
            )

        order = _rank_indices(scores, minimize=minimize)
        label_by_rank = [labels[i] for i in order]
        problem_any = any(bool(value) for value in labels)

        total_candidates += n
        pooled_labels.extend(labels)
        pooled_scores.extend(scores)
        selected_success.append(bool(labels[chosen_index]))
        any_correct.append(problem_any)
        reciprocal_ranks.append(_reciprocal_rank(label_by_rank))

        for k in top_ks_clean:
            hit_by_k[k].append(any(bool(value) for value in label_by_rank[:k]))
            ndcg_by_k[k].append(_ndcg_at_k(label_by_rank, k))

        problem_auc = _pair_auc(labels, scores, minimize=minimize)
        if problem_auc is not None:
            within_auc_values.append(problem_auc)

        if problem_any and not bool(labels[chosen_index]):
            correct_indices = [i for i, label in enumerate(labels) if label]
            best_correct_index = min(
                correct_indices,
                key=lambda i: (
                    scores[i] if minimize else -scores[i],
                    i,
                ),
            )
            rank_lookup = {idx: rank for rank, idx in enumerate(order, start=1)}
            top = order[: max(1, int(failure_top_k))]
            failures.append(
                FailureCase(
                    problem_id=problem_id,
                    chosen_index=int(chosen_index),
                    chosen_score=float(scores[chosen_index]),
                    best_correct_index=int(best_correct_index),
                    best_correct_rank=int(rank_lookup[best_correct_index]),
                    best_correct_score=float(scores[best_correct_index]),
                    score_gap=float(
                        _oriented_score(scores[chosen_index], minimize=minimize)
                        - _oriented_score(scores[best_correct_index], minimize=minimize)
                    ),
                    n_candidates=int(n),
                    top_indices=[int(i) for i in top],
                    top_scores=[float(scores[i]) for i in top],
                )
            )

    if not selected_success:
        raise ValueError("no non-empty scored candidate sets found")

    n_problems = len(selected_success)
    n_any = sum(1 for value in any_correct if value)
    selected_given_any = sum(
        1 for selected, reachable in zip(selected_success, any_correct) if selected and reachable
    )

    failures.sort(
        key=lambda item: (-float(item.score_gap), int(item.best_correct_rank), item.problem_id)
    )

    return {
        "dataset": {
            "problems": int(n_problems),
            "overlap_problem_ids": int(len(problem_ids)),
            "skipped_empty_candidate_sets": int(skipped_empty),
            "skipped_missing_scores": int(skipped_missing_scores),
            "total_candidates": int(total_candidates),
            "avg_candidates_per_problem": total_candidates / float(n_problems),
            "has_any_correct": proportion_summary(n_any, n_problems).to_json_dict(),
        },
        "selection": {
            "selected_correct": proportion_summary(
                sum(1 for value in selected_success if value), n_problems
            ).to_json_dict(),
            "selected_correct_given_any": proportion_summary(
                selected_given_any, n_any
            ).to_json_dict(),
        },
        "ranking": {
            "mean_reciprocal_rank": _mean(reciprocal_ranks),
            "pooled_candidate_auc": _binary_auc(pooled_labels, pooled_scores, minimize=minimize),
            "mean_within_problem_pair_auc": _mean(within_auc_values),
            "top_k_hit": {
                str(k): proportion_summary(
                    sum(1 for value in values if value), len(values)
                ).to_json_dict()
                for k, values in hit_by_k.items()
            },
            "mean_ndcg_at_k": {
                str(k): _mean(values) for k, values in ndcg_by_k.items()
            },
        },
        "failures": {
            "reachable_misses": int(len(failures)),
            "top": [case.to_json_dict() for case in failures[: max(0, int(failure_top_k))]],
        },
        "config": {
            "score_key": str(score_key),
            "chosen_index_key": str(chosen_index_key),
            "minimize": bool(minimize),
            "top_ks": [int(k) for k in top_ks_clean],
        },
    }


def format_markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    selection = summary["selection"]
    ranking = summary["ranking"]
    failures = summary["failures"]

    lines = [
        "# NLVerifier score diagnostics",
        "",
        f"- Problems: {dataset['problems']}",
        f"- Total candidates: {dataset['total_candidates']}",
        f"- Avg candidates/problem: {dataset['avg_candidates_per_problem']:.2f}",
        "- Any-correct ceiling: "
        f"{_pct(dataset['has_any_correct']['rate'])} "
        f"({dataset['has_any_correct']['successes']}/{dataset['has_any_correct']['total']})",
        "- Selected correct: "
        f"{_pct(selection['selected_correct']['rate'])} "
        f"({selection['selected_correct']['successes']}/{selection['selected_correct']['total']})",
        "- Selected correct given any: "
        f"{_pct(selection['selected_correct_given_any']['rate'])} "
        f"({selection['selected_correct_given_any']['successes']}/"
        f"{selection['selected_correct_given_any']['total']})",
        "",
        "## Ranking Quality",
        "",
        f"- Mean reciprocal rank: {ranking['mean_reciprocal_rank']:.4f}",
        f"- Pooled candidate AUC: {_fmt_optional(ranking['pooled_candidate_auc'])}",
        f"- Mean within-problem pair AUC: {_fmt_optional(ranking['mean_within_problem_pair_auc'])}",
        "",
        "| k | top-k contains a correct candidate | mean NDCG@k |",
        "| ---: | ---: | ---: |",
    ]
    for k, hit in ranking["top_k_hit"].items():
        ndcg = ranking["mean_ndcg_at_k"][k]
        lines.append(
            f"| {k} | {_pct(hit['rate'])} ({hit['successes']}/{hit['total']}) | {ndcg:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Reachable Misses",
            "",
            f"- Misses with at least one correct candidate: {failures['reachable_misses']}",
            "",
        ]
    )
    top_failures = failures["top"]
    if top_failures:
        lines.extend(
            [
                "| problem_id | selected | best correct | correct rank | score gap |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for case in top_failures:
            lines.append(
                f"| {case['problem_id']} | {case['chosen_index']} | "
                f"{case['best_correct_index']} | {case['best_correct_rank']} | "
                f"{case['score_gap']:.4f} |"
            )
        lines.append("")
    else:
        lines.append("No reachable misses found.")
        lines.append("")

    return "\n".join(lines)


def write_failures_jsonl(path: str | Path, summary: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for case in summary["failures"]["top"]:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")


def _chosen_index(record: dict[str, Any], *, chosen_index_key: str) -> int:
    if chosen_index_key in record:
        return int(record[chosen_index_key])
    raw = record.get("chosen_indices")
    if isinstance(raw, list) and raw:
        return int(raw[0])
    raise ValueError(f"selection record has no {chosen_index_key!r}/chosen_indices: {record!r}")


def _as_finite_float(value: Any, *, problem_id: str) -> float:
    out = float(value)
    if not isfinite(out):
        raise ValueError(f"non-finite score for problem_id={problem_id!r}: {value!r}")
    return out


def _rank_indices(scores: list[float], *, minimize: bool) -> list[int]:
    return sorted(
        range(len(scores)),
        key=lambda i: (_oriented_score(scores[i], minimize=minimize) * -1.0, i),
    )


def _oriented_score(score: float, *, minimize: bool) -> float:
    return -float(score) if minimize else float(score)


def _reciprocal_rank(label_by_rank: list[int]) -> float:
    for rank, label in enumerate(label_by_rank, start=1):
        if label:
            return 1.0 / float(rank)
    return 0.0


def _ndcg_at_k(label_by_rank: list[int], k: int) -> float:
    k = max(1, int(k))
    dcg = 0.0
    for rank, label in enumerate(label_by_rank[:k], start=1):
        if label:
            dcg += 1.0 / log2(rank + 1.0)
    ideal_hits = min(sum(1 for label in label_by_rank if label), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def _pair_auc(labels: list[int], scores: list[float], *, minimize: bool) -> float | None:
    positives = [
        _oriented_score(score, minimize=minimize)
        for label, score in zip(labels, scores)
        if label
    ]
    negatives = [
        _oriented_score(score, minimize=minimize)
        for label, score in zip(labels, scores)
        if not label
    ]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / float(total)


def _binary_auc(labels: list[int], scores: list[float], *, minimize: bool) -> float | None:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    values = [
        (_oriented_score(score, minimize=minimize), int(label))
        for label, score in zip(labels, scores)
    ]
    n_pos = sum(1 for _score, label in values if label)
    n_neg = len(values) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    values.sort(key=lambda item: item[0])
    rank_sum_pos = 0.0
    rank = 1
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[j][0] == values[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        rank_sum_pos += avg_rank * sum(1 for _score, label in values[i:j] if label)
        rank += j - i
        i = j

    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(value) for value in values]
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _parse_top_ks(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        out.append(int(text))
    return out or [1, 2, 5, 10]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--selections", required=True)
    parser.add_argument("--score-key", default="scores")
    parser.add_argument("--chosen-index-key", default="chosen_index")
    parser.add_argument("--top-ks", default="1,2,5,10")
    parser.add_argument("--minimize", action="store_true", help="Treat lower scores as better.")
    parser.add_argument("--failure-top-k", type=int, default=10)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    parser.add_argument("--failures-jsonl")
    args = parser.parse_args()

    summary = analyze_scores(
        candidates_path=args.candidates,
        selections_path=args.selections,
        score_key=str(args.score_key),
        chosen_index_key=str(args.chosen_index_key),
        top_ks=_parse_top_ks(str(args.top_ks)),
        minimize=bool(args.minimize),
        failure_top_k=int(args.failure_top_k),
    )
    markdown = format_markdown(summary)
    print(markdown)

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_md:
        Path(args.output_md).write_text(markdown + "\n", encoding="utf-8")
    if args.failures_jsonl:
        write_failures_jsonl(args.failures_jsonl, summary)


if __name__ == "__main__":
    main()
