"""Selective prediction analysis for NLVerifier.

This module answers a deployment-oriented question: if NLVerifier is allowed to
abstain on low-confidence problems, what accuracy can it achieve at each
coverage level?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import exp, isfinite
from pathlib import Path
from typing import Any

from .jsonl import matching_problem_ids
from .nlverifier_diagnostics import load_jsonl_map
from .schema import validate_grouped_candidates


@dataclass(frozen=True)
class SelectiveExample:
    problem_id: str
    selected_correct: bool
    has_any_correct: bool
    chosen_index: int
    chosen_score: float
    chosen_probability: float
    runner_up_index: int | None
    runner_up_score: float | None
    runner_up_probability: float | None
    score_margin: float | None
    probability_margin: float | None
    n_candidates: int

    def confidence_value(self, key: str) -> float:
        value = getattr(self, key)
        if value is None:
            return 0.0
        return float(value)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "selected_correct": self.selected_correct,
            "has_any_correct": self.has_any_correct,
            "chosen_index": self.chosen_index,
            "chosen_score": self.chosen_score,
            "chosen_probability": self.chosen_probability,
            "runner_up_index": self.runner_up_index,
            "runner_up_score": self.runner_up_score,
            "runner_up_probability": self.runner_up_probability,
            "score_margin": self.score_margin,
            "probability_margin": self.probability_margin,
            "n_candidates": self.n_candidates,
        }


def load_selective_examples(
    *,
    scored_path: str | Path | None = None,
    candidates_path: str | Path | None = None,
    selections_path: str | Path | None = None,
    calibration_json: str | Path | None = None,
    temperature: float = 1.0,
    score_key: str = "scores",
    minimize: bool = False,
    allow_partial_overlap: bool = False,
) -> list[SelectiveExample]:
    if scored_path is None and (candidates_path is None or selections_path is None):
        raise ValueError("provide either scored_path or both candidates_path and selections_path")

    score_temperature = _load_temperature(calibration_json, temperature)
    if scored_path is not None:
        rows = load_jsonl_map(scored_path)
        examples = [
            _example_from_rows(
                problem_id=problem_id,
                candidate_row=row,
                selection_row=row,
                score_key=score_key,
                temperature=score_temperature,
                minimize=minimize,
            )
            for problem_id, row in sorted(rows.items())
        ]
    else:
        assert candidates_path is not None
        assert selections_path is not None
        candidates = load_jsonl_map(candidates_path)
        selections = load_jsonl_map(selections_path)
        problem_ids = matching_problem_ids(
            candidates,
            selections,
            left_name="candidates",
            right_name="selections",
            allow_partial_overlap=allow_partial_overlap,
        )
        examples = [
            _example_from_rows(
                problem_id=problem_id,
                candidate_row=candidates[problem_id],
                selection_row=selections[problem_id],
                score_key=score_key,
                temperature=score_temperature,
                minimize=minimize,
            )
            for problem_id in problem_ids
        ]
    return [example for example in examples if example.n_candidates > 0]


def analyze_selective_risk(
    examples: list[SelectiveExample],
    *,
    confidence_key: str = "chosen_probability",
    min_coverages: list[float] | None = None,
    top_failures: int = 15,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("no selective examples found")
    if confidence_key not in {
        "chosen_probability",
        "probability_margin",
        "score_margin",
    }:
        raise ValueError(f"unsupported confidence_key={confidence_key!r}")

    ordered = sorted(
        examples,
        key=lambda item: (-item.confidence_value(confidence_key), item.problem_id),
    )
    n = len(ordered)
    points: list[dict[str, Any]] = []
    for k in range(1, n + 1):
        accepted = ordered[:k]
        correct = sum(1 for item in accepted if item.selected_correct)
        points.append(
            {
                "accepted": int(k),
                "coverage": k / float(n),
                "selected_correct": int(correct),
                "accuracy": correct / float(k),
                "risk": 1.0 - (correct / float(k)),
                "threshold": float(accepted[-1].confidence_value(confidence_key)),
            }
        )

    targets = min_coverages or [0.25, 0.5, 0.75, 1.0]
    coverage_table = []
    for target in targets:
        target = min(1.0, max(0.0, float(target)))
        needed = max(1, int(_ceil(target * n)))
        coverage_table.append(points[min(needed, n) - 1])

    best_accuracy = max(points, key=lambda row: (row["accuracy"], row["coverage"]))
    highest_confidence_errors = [
        item.to_json_dict()
        for item in ordered
        if not item.selected_correct
    ][: max(0, int(top_failures))]
    any_correct = sum(1 for item in examples if item.has_any_correct)
    total_correct = sum(1 for item in examples if item.selected_correct)

    return {
        "dataset": {
            "problems": int(n),
            "has_any_correct": int(any_correct),
            "has_any_correct_rate": any_correct / float(n),
            "selected_correct": int(total_correct),
            "selected_accuracy": total_correct / float(n),
        },
        "confidence_key": str(confidence_key),
        "coverage_table": coverage_table,
        "best_accuracy_prefix": best_accuracy,
        "risk_coverage_curve": points,
        "high_confidence_errors": highest_confidence_errors,
    }


def format_markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    lines = [
        "# NLVerifier Selective Risk",
        "",
        f"- Problems: {dataset['problems']}",
        f"- Any-correct ceiling: {_pct(dataset['has_any_correct_rate'])} ({dataset['has_any_correct']}/{dataset['problems']})",
        f"- Full-coverage selected accuracy: {_pct(dataset['selected_accuracy'])} ({dataset['selected_correct']}/{dataset['problems']})",
        f"- Confidence key: `{summary['confidence_key']}`",
        "",
        "## Coverage Table",
        "",
        "| accepted | coverage | accuracy | risk | threshold |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["coverage_table"]:
        lines.append(
            f"| {row['accepted']} | {_pct(row['coverage'])} | {_pct(row['accuracy'])} | "
            f"{_pct(row['risk'])} | {row['threshold']:.4f} |"
        )

    best = summary["best_accuracy_prefix"]
    lines.extend(
        [
            "",
            "## Best Accuracy Prefix",
            "",
            f"- Accepted: {best['accepted']}",
            f"- Coverage: {_pct(best['coverage'])}",
            f"- Accuracy: {_pct(best['accuracy'])}",
            f"- Threshold: `{best['threshold']:.4f}`",
            "",
            "## Highest-Confidence Errors",
            "",
        ]
    )
    errors = summary["high_confidence_errors"]
    if errors:
        lines.extend(
            [
                "| problem_id | chosen | probability | probability margin | score margin |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in errors:
            lines.append(
                f"| {_md(row['problem_id'])} | {row['chosen_index']} | "
                f"{row['chosen_probability']:.4f} | {_fmt_optional(row['probability_margin'])} | "
                f"{_fmt_optional(row['score_margin'])} |"
            )
    else:
        lines.append("No selected errors.")
    lines.append("")
    return "\n".join(lines)


def _example_from_rows(
    *,
    problem_id: str,
    candidate_row: dict[str, Any],
    selection_row: dict[str, Any],
    score_key: str,
    temperature: float,
    minimize: bool,
) -> SelectiveExample:
    grouped = validate_grouped_candidates(candidate_row, require_labels=True)
    assert grouped.labels is not None
    labels = [1 if int(label) else 0 for label in grouped.labels]
    n = len(grouped.candidates)
    if n == 0:
        return SelectiveExample(
            problem_id=str(problem_id),
            selected_correct=False,
            has_any_correct=False,
            chosen_index=0,
            chosen_score=0.0,
            chosen_probability=0.0,
            runner_up_index=None,
            runner_up_score=None,
            runner_up_probability=None,
            score_margin=None,
            probability_margin=None,
            n_candidates=0,
        )

    raw_scores = selection_row.get(score_key)
    if not isinstance(raw_scores, list):
        raise ValueError(f"{score_key!r} must be a list for problem_id={problem_id!r}")
    if len(raw_scores) != n:
        raise ValueError(f"score length mismatch for {problem_id}: {len(raw_scores)} vs {n}")
    scores = [_as_finite_float(value, problem_id=problem_id) for value in raw_scores]
    chosen_index = int(selection_row.get("chosen_index", _best_index(scores, minimize=minimize)))
    if chosen_index < 0 or chosen_index >= n:
        raise ValueError(f"chosen_index out of range for {problem_id}: {chosen_index} (n={n})")

    ranked = sorted(range(n), key=lambda i: (-_oriented(scores[i], minimize=minimize), i))
    runner_up = next((idx for idx in ranked if idx != chosen_index), None)
    chosen_oriented = _oriented(scores[chosen_index], minimize=minimize)
    chosen_prob = _sigmoid(chosen_oriented / temperature)
    if runner_up is None:
        runner_score = None
        runner_prob = None
        score_margin = None
        probability_margin = None
    else:
        runner_score = scores[runner_up]
        runner_oriented = _oriented(runner_score, minimize=minimize)
        runner_prob = _sigmoid(runner_oriented / temperature)
        score_margin = chosen_oriented - runner_oriented
        probability_margin = chosen_prob - runner_prob

    return SelectiveExample(
        problem_id=str(problem_id),
        selected_correct=bool(labels[chosen_index]),
        has_any_correct=any(bool(label) for label in labels),
        chosen_index=int(chosen_index),
        chosen_score=float(scores[chosen_index]),
        chosen_probability=float(chosen_prob),
        runner_up_index=int(runner_up) if runner_up is not None else None,
        runner_up_score=float(runner_score) if runner_score is not None else None,
        runner_up_probability=float(runner_prob) if runner_prob is not None else None,
        score_margin=float(score_margin) if score_margin is not None else None,
        probability_margin=float(probability_margin) if probability_margin is not None else None,
        n_candidates=int(n),
    )


def _load_temperature(path: str | Path | None, fallback: float) -> float:
    temperature = float(fallback)
    if path is not None and str(path).strip():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "temperature" in payload and isinstance(payload["temperature"], dict):
            temperature = float(payload["temperature"].get("fitted", payload["temperature"].get("input", fallback)))
        else:
            temperature = float(payload.get("temperature", payload.get("fitted_temperature", fallback)))
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be positive and finite, got {temperature!r}")
    return temperature


def _best_index(scores: list[float], *, minimize: bool) -> int:
    if minimize:
        return min(range(len(scores)), key=lambda i: (scores[i], i))
    return max(range(len(scores)), key=lambda i: (scores[i], -i))


def _oriented(score: float, *, minimize: bool) -> float:
    return -float(score) if minimize else float(score)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def _as_finite_float(value: Any, *, problem_id: str) -> float:
    out = float(value)
    if not isfinite(out):
        raise ValueError(f"non-finite score for problem_id={problem_id!r}: {value!r}")
    return out


def _ceil(value: float) -> int:
    whole = int(value)
    return whole if whole == value else whole + 1


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _fmt_optional(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _md(value: object) -> str:
    return str(value).replace("|", "\\|")


def _parse_coverages(raw: str) -> list[float]:
    out: list[float] = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            out.append(float(text))
    return out or [0.25, 0.5, 0.75, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", help="Scored grouped JSONL containing labels, scores, and chosen_index.")
    parser.add_argument("--candidates", help="Grouped candidates JSONL with labels.")
    parser.add_argument("--selections", help="Selection JSONL containing chosen_index and scores.")
    parser.add_argument("--score-key", default="scores")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--calibration-json", default="")
    parser.add_argument("--minimize", action="store_true")
    parser.add_argument(
        "--allow-partial-overlap",
        action="store_true",
        help="Analyze only overlapping candidates/selections IDs instead of failing on mismatches.",
    )
    parser.add_argument(
        "--confidence-key",
        default="chosen_probability",
        choices=("chosen_probability", "probability_margin", "score_margin"),
    )
    parser.add_argument("--coverages", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--top-failures", type=int, default=15)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    examples = load_selective_examples(
        scored_path=args.scores,
        candidates_path=args.candidates,
        selections_path=args.selections,
        calibration_json=str(args.calibration_json).strip() or None,
        temperature=float(args.temperature),
        score_key=str(args.score_key),
        minimize=bool(args.minimize),
        allow_partial_overlap=bool(args.allow_partial_overlap),
    )
    summary = analyze_selective_risk(
        examples,
        confidence_key=str(args.confidence_key),
        min_coverages=_parse_coverages(str(args.coverages)),
        top_failures=int(args.top_failures),
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


if __name__ == "__main__":
    main()
