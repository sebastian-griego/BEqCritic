"""Reliability and temperature-scaling diagnostics for NLVerifier scores.

NLVerifier emits raw scalar scores for NL/Lean pairs. This module evaluates how
well those scores behave as calibrated probabilities after a sigmoid transform,
optionally fitting one scalar temperature on labeled grouped-candidate data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
from pathlib import Path
from typing import Any, Iterable

from .jsonl import matching_problem_ids
from .nlverifier_diagnostics import load_jsonl_map
from .schema import validate_grouped_candidates


@dataclass(frozen=True)
class ScoredExample:
    problem_id: str
    candidate_index: int
    label: int
    score: float

    def oriented_score(self, *, minimize: bool) -> float:
        return -self.score if minimize else self.score


def load_scored_examples(
    *,
    scored_path: str | Path | None = None,
    candidates_path: str | Path | None = None,
    selections_path: str | Path | None = None,
    score_key: str = "scores",
    minimize: bool = False,
    allow_partial_overlap: bool = False,
) -> list[ScoredExample]:
    """Load labeled candidate scores from a scored JSONL or joined JSONLs."""

    if scored_path is None and (candidates_path is None or selections_path is None):
        raise ValueError("provide either scored_path or both candidates_path and selections_path")

    if scored_path is not None:
        rows = load_jsonl_map(scored_path)
        examples: list[ScoredExample] = []
        for problem_id, row in sorted(rows.items()):
            examples.extend(
                _examples_from_joined_row(
                    problem_id=problem_id,
                    candidate_row=row,
                    score_row=row,
                    score_key=score_key,
                    minimize=minimize,
                )
            )
        return examples

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

    examples = []
    for problem_id in problem_ids:
        examples.extend(
            _examples_from_joined_row(
                problem_id=problem_id,
                candidate_row=candidates[problem_id],
                score_row=selections[problem_id],
                score_key=score_key,
                minimize=minimize,
            )
        )
    return examples


def analyze_calibration(
    examples: Iterable[ScoredExample],
    *,
    bins: int = 10,
    temperature: float = 1.0,
    fit_temperature: bool = False,
    minimize: bool = False,
) -> dict[str, Any]:
    examples = list(examples)
    if not examples:
        raise ValueError("no scored examples found")

    labels = [int(example.label) for example in examples]
    scores = [example.oriented_score(minimize=minimize) for example in examples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"need both positive and negative labels; got n_pos={n_pos} n_neg={n_neg}")

    base_temperature = _validate_temperature(temperature)
    fitted_temperature = (
        fit_temperature_for_nll(scores, labels) if fit_temperature else base_temperature
    )
    before = calibration_metrics(scores, labels, temperature=base_temperature, bins=bins)
    after = calibration_metrics(scores, labels, temperature=fitted_temperature, bins=bins)
    threshold = threshold_sweep(scores, labels, temperature=fitted_temperature)

    problem_count = len({example.problem_id for example in examples})
    return {
        "dataset": {
            "problems": int(problem_count),
            "candidates": int(len(examples)),
            "positive": int(n_pos),
            "negative": int(n_neg),
            "positive_rate": n_pos / float(len(examples)),
        },
        "temperature": {
            "input": float(base_temperature),
            "fitted": float(fitted_temperature),
            "fit_temperature": bool(fit_temperature),
        },
        "before": before,
        "after": after,
        "thresholds": threshold,
        "config": {
            "bins": int(bins),
            "minimize": bool(minimize),
        },
    }


def calibration_metrics(
    scores: list[float],
    labels: list[int],
    *,
    temperature: float,
    bins: int,
) -> dict[str, Any]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    temperature = _validate_temperature(temperature)
    bins = max(1, int(bins))
    probs = [_sigmoid(score / temperature) for score in scores]
    nll = binary_nll_from_probs(probs, labels)
    brier = sum((prob - label) ** 2 for prob, label in zip(probs, labels)) / len(probs)
    reliability = reliability_bins(probs, labels, bins=bins)
    ece = sum((bucket["count"] / len(probs)) * bucket["abs_gap"] for bucket in reliability)
    mce = max((bucket["abs_gap"] for bucket in reliability), default=0.0)
    return {
        "temperature": float(temperature),
        "nll": float(nll),
        "brier": float(brier),
        "ece": float(ece),
        "mce": float(mce),
        "reliability_bins": reliability,
    }


def fit_temperature_for_nll(
    scores: list[float],
    labels: list[int],
    *,
    min_temperature: float = 0.05,
    max_temperature: float = 20.0,
    iterations: int = 80,
) -> float:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        raise ValueError("cannot fit temperature without scores")

    lo = log(_validate_temperature(min_temperature))
    hi = log(_validate_temperature(max_temperature))
    inv_phi = (sqrt(5.0) - 1.0) / 2.0
    inv_phi_sq = (3.0 - sqrt(5.0)) / 2.0

    c = lo + inv_phi_sq * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    fc = _nll_at_log_temperature(scores, labels, c)
    fd = _nll_at_log_temperature(scores, labels, d)
    for _ in range(max(1, int(iterations))):
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = lo + inv_phi_sq * (hi - lo)
            fc = _nll_at_log_temperature(scores, labels, c)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + inv_phi * (hi - lo)
            fd = _nll_at_log_temperature(scores, labels, d)
    return float(exp((lo + hi) / 2.0))


def binary_nll_from_probs(probs: list[float], labels: list[int]) -> float:
    eps = 1e-12
    total = 0.0
    for prob, label in zip(probs, labels):
        p = min(1.0 - eps, max(eps, float(prob)))
        if int(label):
            total -= log(p)
        else:
            total -= log(1.0 - p)
    return total / float(len(probs))


def reliability_bins(probs: list[float], labels: list[int], *, bins: int) -> list[dict[str, Any]]:
    bins = max(1, int(bins))
    rows: list[dict[str, Any]] = []
    for idx in range(bins):
        lo = idx / float(bins)
        hi = (idx + 1) / float(bins)
        bucket_indices = [
            i
            for i, prob in enumerate(probs)
            if (lo <= prob < hi) or (idx == bins - 1 and prob == 1.0)
        ]
        if not bucket_indices:
            rows.append(
                {
                    "bin": int(idx),
                    "lower": float(lo),
                    "upper": float(hi),
                    "count": 0,
                    "avg_confidence": None,
                    "accuracy": None,
                    "gap": None,
                    "abs_gap": 0.0,
                }
            )
            continue
        avg_conf = sum(probs[i] for i in bucket_indices) / len(bucket_indices)
        acc = sum(int(labels[i]) for i in bucket_indices) / len(bucket_indices)
        gap = avg_conf - acc
        rows.append(
            {
                "bin": int(idx),
                "lower": float(lo),
                "upper": float(hi),
                "count": int(len(bucket_indices)),
                "avg_confidence": float(avg_conf),
                "accuracy": float(acc),
                "gap": float(gap),
                "abs_gap": abs(float(gap)),
            }
        )
    return rows


def threshold_sweep(
    scores: list[float],
    labels: list[int],
    *,
    temperature: float,
) -> dict[str, Any]:
    probs = [_sigmoid(score / _validate_temperature(temperature)) for score in scores]
    thresholds = sorted({i / 100.0 for i in range(1, 100)} | set(probs))
    best_f1: dict[str, Any] | None = None
    best_accuracy: dict[str, Any] | None = None
    for threshold in thresholds:
        row = _threshold_metrics(probs, labels, threshold)
        if best_f1 is None or (
            row["f1"],
            row["precision"],
            row["recall"],
            -row["threshold"],
        ) > (
            best_f1["f1"],
            best_f1["precision"],
            best_f1["recall"],
            -best_f1["threshold"],
        ):
            best_f1 = row
        if best_accuracy is None or (
            row["accuracy"],
            row["f1"],
            -row["threshold"],
        ) > (
            best_accuracy["accuracy"],
            best_accuracy["f1"],
            -best_accuracy["threshold"],
        ):
            best_accuracy = row

    assert best_f1 is not None
    assert best_accuracy is not None
    return {
        "best_f1": best_f1,
        "best_accuracy": best_accuracy,
        "at_0_5": _threshold_metrics(probs, labels, 0.5),
    }


def format_markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    before = summary["before"]
    after = summary["after"]
    thresholds = summary["thresholds"]
    lines = [
        "# NLVerifier Calibration",
        "",
        f"- Problems: {dataset['problems']}",
        f"- Candidate scores: {dataset['candidates']}",
        f"- Positives: {dataset['positive']} ({_pct(dataset['positive_rate'])})",
        f"- Fitted temperature: `{summary['temperature']['fitted']:.4f}`",
        "",
        "## Metrics",
        "",
        "| transform | temperature | NLL | Brier | ECE | MCE |",
        "|---|---:|---:|---:|---:|---:|",
        _metric_row("input", before),
        _metric_row("fitted", after),
        "",
        "## Thresholds",
        "",
        "| objective | threshold | precision | recall | F1 | accuracy | predicted positive |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _threshold_row("at 0.5", thresholds["at_0_5"]),
        _threshold_row("best F1", thresholds["best_f1"]),
        _threshold_row("best accuracy", thresholds["best_accuracy"]),
        "",
        "## Reliability Bins",
        "",
        "| bin | count | avg confidence | empirical accuracy | gap |",
        "|---:|---:|---:|---:|---:|",
    ]
    for bucket in after["reliability_bins"]:
        if not bucket["count"]:
            continue
        lines.append(
            f"| {bucket['lower']:.1f}-{bucket['upper']:.1f} | {bucket['count']} | "
            f"{bucket['avg_confidence']:.3f} | {bucket['accuracy']:.3f} | {bucket['gap']:+.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _examples_from_joined_row(
    *,
    problem_id: str,
    candidate_row: dict[str, Any],
    score_row: dict[str, Any],
    score_key: str,
    minimize: bool,
) -> list[ScoredExample]:
    del minimize
    grouped = validate_grouped_candidates(candidate_row, require_labels=True)
    assert grouped.labels is not None
    raw_scores = score_row.get(score_key)
    if not isinstance(raw_scores, list):
        raise ValueError(f"{score_key!r} must be a list for problem_id={problem_id!r}")
    if len(raw_scores) != len(grouped.candidates):
        raise ValueError(
            f"score length mismatch for {problem_id}: {len(raw_scores)} vs {len(grouped.candidates)}"
        )
    out: list[ScoredExample] = []
    for idx, (label, score) in enumerate(zip(grouped.labels, raw_scores)):
        score_float = float(score)
        if not isfinite(score_float):
            raise ValueError(f"non-finite score for problem_id={problem_id!r}: {score!r}")
        out.append(
            ScoredExample(
                problem_id=str(problem_id),
                candidate_index=int(idx),
                label=1 if int(label) else 0,
                score=score_float,
            )
        )
    return out


def _nll_at_log_temperature(scores: list[float], labels: list[int], log_temperature: float) -> float:
    temperature = exp(log_temperature)
    probs = [_sigmoid(score / temperature) for score in scores]
    return binary_nll_from_probs(probs, labels)


def _threshold_metrics(probs: list[float], labels: list[int], threshold: float) -> dict[str, Any]:
    pred = [prob >= threshold for prob in probs]
    tp = sum(1 for y, p in zip(labels, pred) if int(y) == 1 and p)
    fp = sum(1 for y, p in zip(labels, pred) if int(y) == 0 and p)
    tn = sum(1 for y, p in zip(labels, pred) if int(y) == 0 and not p)
    fn = sum(1 for y, p in zip(labels, pred) if int(y) == 1 and not p)
    precision = tp / float(tp + fp) if (tp + fp) else 0.0
    recall = tp / float(tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / float(len(labels)) if labels else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "predicted_positive": int(tp + fp),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


def _validate_temperature(value: float) -> float:
    temperature = float(value)
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be a positive finite value, got {value!r}")
    return temperature


def _metric_row(name: str, row: dict[str, Any]) -> str:
    return (
        f"| {name} | {row['temperature']:.4f} | {row['nll']:.4f} | "
        f"{row['brier']:.4f} | {row['ece']:.4f} | {row['mce']:.4f} |"
    )


def _threshold_row(name: str, row: dict[str, Any]) -> str:
    return (
        f"| {name} | {row['threshold']:.3f} | {_pct(row['precision'])} | "
        f"{_pct(row['recall'])} | {_pct(row['f1'])} | {_pct(row['accuracy'])} | "
        f"{row['predicted_positive']} |"
    )


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", help="Scored grouped JSONL containing labels and per-candidate scores.")
    parser.add_argument("--candidates", help="Grouped candidates JSONL with labels.")
    parser.add_argument("--selections", help="Selection JSONL containing per-candidate scores.")
    parser.add_argument("--score-key", default="scores")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--fit-temperature", action="store_true")
    parser.add_argument("--minimize", action="store_true", help="Treat lower raw scores as better.")
    parser.add_argument(
        "--allow-partial-overlap",
        action="store_true",
        help="Analyze only overlapping candidates/selections IDs instead of failing on mismatches.",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    examples = load_scored_examples(
        scored_path=args.scores,
        candidates_path=args.candidates,
        selections_path=args.selections,
        score_key=str(args.score_key),
        minimize=bool(args.minimize),
        allow_partial_overlap=bool(args.allow_partial_overlap),
    )
    summary = analyze_calibration(
        examples,
        bins=int(args.bins),
        temperature=float(args.temperature),
        fit_temperature=bool(args.fit_temperature),
        minimize=bool(args.minimize),
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
