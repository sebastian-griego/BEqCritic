"""Compare confidence signals for NLVerifier selective prediction.

The selective-risk report answers how one confidence score behaves when the
reranker abstains. This audit compares the available confidence scores side by
side so it is clear which signal is useful for coverage/accuracy tradeoffs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .nlverifier_selective import (
    SelectiveExample,
    analyze_selective_risk,
    load_selective_examples,
)


DEFAULT_CONFIDENCE_KEYS = ("chosen_probability", "probability_margin", "score_margin")
DEFAULT_COVERAGES = (0.25, 0.5, 0.75, 1.0)


def analyze_confidence_signals(
    examples: Iterable[SelectiveExample],
    *,
    confidence_keys: Iterable[str] = DEFAULT_CONFIDENCE_KEYS,
    coverages: Iterable[float] = DEFAULT_COVERAGES,
) -> dict[str, Any]:
    examples = list(examples)
    if not examples:
        raise ValueError("no selective examples found")

    keys = [str(key) for key in confidence_keys]
    if not keys:
        raise ValueError("at least one confidence key is required")
    coverage_targets = [min(1.0, max(0.0, float(value))) for value in coverages]
    if not coverage_targets:
        coverage_targets = list(DEFAULT_COVERAGES)

    per_key: dict[str, Any] = {}
    for key in keys:
        selective = analyze_selective_risk(
            examples,
            confidence_key=key,
            min_coverages=coverage_targets,
            top_failures=0,
        )
        curve = selective["risk_coverage_curve"]
        mean_risk = sum(float(row["risk"]) for row in curve) / float(len(curve))
        mean_accuracy = sum(float(row["accuracy"]) for row in curve) / float(len(curve))
        per_key[key] = {
            "dataset": selective["dataset"],
            "coverage_table": selective["coverage_table"],
            "best_accuracy_prefix": selective["best_accuracy_prefix"],
            "mean_prefix_risk": float(mean_risk),
            "mean_prefix_accuracy": float(mean_accuracy),
        }

    coverage_comparison = _coverage_comparison(per_key, coverage_targets)
    return {
        "dataset": per_key[keys[0]]["dataset"],
        "confidence_keys": keys,
        "coverage_targets": coverage_targets,
        "signals": per_key,
        "coverage_comparison": coverage_comparison,
        "best_by_mean_prefix_risk": min(keys, key=lambda key: (per_key[key]["mean_prefix_risk"], key)),
        "best_by_mean_prefix_accuracy": max(
            keys,
            key=lambda key: (
                per_key[key]["mean_prefix_accuracy"],
                -per_key[key]["mean_prefix_risk"],
                key,
            ),
        ),
    }


def format_markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    keys = [str(key) for key in summary["confidence_keys"]]
    lines = [
        "# NLVerifier Confidence Signal Audit",
        "",
        f"- Problems: {dataset['problems']}",
        f"- Any-correct ceiling: {_pct(dataset['has_any_correct_rate'])} "
        f"({dataset['has_any_correct']}/{dataset['problems']})",
        f"- Full-coverage selected accuracy: {_pct(dataset['selected_accuracy'])} "
        f"({dataset['selected_correct']}/{dataset['problems']})",
        f"- Best mean prefix risk: `{summary['best_by_mean_prefix_risk']}`",
        "",
        "## Signal Summary",
        "",
        "| confidence key | mean prefix risk | mean prefix accuracy | best prefix accuracy | best prefix coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in keys:
        row = summary["signals"][key]
        best = row["best_accuracy_prefix"]
        lines.append(
            f"| `{key}` | {_pct(row['mean_prefix_risk'])} | "
            f"{_pct(row['mean_prefix_accuracy'])} | {_pct(best['accuracy'])} | "
            f"{_pct(best['coverage'])} |"
        )

    lines.extend(["", "## Coverage Comparison", ""])
    header = "| target coverage | " + " | ".join(f"`{key}` accuracy" for key in keys) + " | best key |"
    divider = "|---:|" + "|".join("---:" for _ in keys) + "|---|"
    lines.extend([header, divider])
    for row in summary["coverage_comparison"]:
        accuracies = " | ".join(_pct(row["by_key"][key]["accuracy"]) for key in keys)
        lines.append(f"| {_pct(row['target_coverage'])} | {accuracies} | `{row['best_key']}` |")

    lines.extend(["", "## Coverage Details", ""])
    for key in keys:
        lines.extend(
            [
                f"### {key}",
                "",
                "| accepted | coverage | accuracy | risk | threshold |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["signals"][key]["coverage_table"]:
            lines.append(
                f"| {row['accepted']} | {_pct(row['coverage'])} | {_pct(row['accuracy'])} | "
                f"{_pct(row['risk'])} | {row['threshold']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _coverage_comparison(
    per_key: dict[str, Any],
    coverage_targets: list[float],
) -> list[dict[str, Any]]:
    keys = list(per_key)
    rows = []
    for idx, target in enumerate(coverage_targets):
        by_key: dict[str, Any] = {}
        for key in keys:
            row = per_key[key]["coverage_table"][idx]
            by_key[key] = {
                "accepted": int(row["accepted"]),
                "coverage": float(row["coverage"]),
                "accuracy": float(row["accuracy"]),
                "risk": float(row["risk"]),
                "threshold": float(row["threshold"]),
            }
        best_key = max(
            enumerate(keys),
            key=lambda item: (
                by_key[item[1]]["accuracy"],
                -by_key[item[1]]["risk"],
                -item[0],
            ),
        )[1]
        rows.append(
            {
                "target_coverage": float(target),
                "best_key": best_key,
                "by_key": by_key,
            }
        )
    return rows


def _parse_floats(raw: str, *, default: Iterable[float]) -> list[float]:
    values = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            values.append(float(text))
    return values or [float(value) for value in default]


def _parse_keys(raw: str) -> list[str]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    return values or list(DEFAULT_CONFIDENCE_KEYS)


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


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
        "--confidence-keys",
        default=",".join(DEFAULT_CONFIDENCE_KEYS),
        help="Comma-separated confidence keys to compare.",
    )
    parser.add_argument("--coverages", default=",".join(str(value) for value in DEFAULT_COVERAGES))
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
    )
    summary = analyze_confidence_signals(
        examples,
        confidence_keys=_parse_keys(str(args.confidence_keys)),
        coverages=_parse_floats(str(args.coverages), default=DEFAULT_COVERAGES),
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
