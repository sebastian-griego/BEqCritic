"""Risk-aware threshold recommendations for NLVerifier abstention.

This module turns a selective-risk curve into deployable threshold candidates.
For each requested target accuracy, it picks the highest-coverage prefix whose
Wilson lower confidence bound still meets the target, and reports a best-effort
fallback when the labeled sample is too small or too noisy to certify it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .nlverifier_selective import SelectiveExample, load_selective_examples
from .statistics import proportion_summary


DEFAULT_CONFIDENCE_KEYS = ("chosen_probability",)
DEFAULT_TARGET_ACCURACIES = (0.5, 0.6, 0.7)


def analyze_thresholds(
    examples: Iterable[SelectiveExample],
    *,
    confidence_keys: Iterable[str] = DEFAULT_CONFIDENCE_KEYS,
    target_accuracies: Iterable[float] = DEFAULT_TARGET_ACCURACIES,
    min_accepted: int = 5,
) -> dict[str, Any]:
    examples = list(examples)
    if not examples:
        raise ValueError("no selective examples found")
    keys = [str(key) for key in confidence_keys]
    if not keys:
        raise ValueError("at least one confidence key is required")
    targets = [min(1.0, max(0.0, float(value))) for value in target_accuracies]
    if not targets:
        targets = list(DEFAULT_TARGET_ACCURACIES)
    min_accepted = max(1, int(min_accepted))

    any_correct = sum(1 for item in examples if item.has_any_correct)
    selected_correct = sum(1 for item in examples if item.selected_correct)
    signals = {
        key: _analyze_key(
            examples,
            confidence_key=key,
            targets=targets,
            min_accepted=min_accepted,
        )
        for key in keys
    }
    return {
        "dataset": {
            "problems": len(examples),
            "has_any_correct": int(any_correct),
            "has_any_correct_rate": any_correct / float(len(examples)),
            "selected_correct": int(selected_correct),
            "selected_accuracy": selected_correct / float(len(examples)),
        },
        "confidence_keys": keys,
        "target_accuracies": targets,
        "min_accepted": min_accepted,
        "signals": signals,
    }


def format_markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    lines = [
        "# NLVerifier Threshold Recommendations",
        "",
        f"- Problems: {dataset['problems']}",
        f"- Any-correct ceiling: {_pct(dataset['has_any_correct_rate'])} "
        f"({dataset['has_any_correct']}/{dataset['problems']})",
        f"- Full-coverage selected accuracy: {_pct(dataset['selected_accuracy'])} "
        f"({dataset['selected_correct']}/{dataset['problems']})",
        f"- Minimum accepted examples: `{summary['min_accepted']}`",
        "",
        "## Recommendations",
        "",
        "| confidence key | target accuracy | certified? | accepted | coverage | accuracy | 95% LCB | threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, signal in summary["signals"].items():
        for row in signal["target_recommendations"]:
            lines.append(
                f"| `{key}` | {_pct(row['target_accuracy'])} | {_yes_no(row['meets_target'])} | "
                f"{row['accepted']} | {_pct(row['coverage'])} | {_pct(row['accuracy'])} | "
                f"{_pct(row['accuracy_ci95']['low'])} | {row['threshold']:.4f} |"
            )

    lines.extend(["", "## Best Lower-Bound Prefixes", ""])
    lines.extend(
        [
            "| confidence key | accepted | coverage | accuracy | 95% LCB | threshold |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, signal in summary["signals"].items():
        row = signal["best_lower_bound_prefix"]
        lines.append(
            f"| `{key}` | {row['accepted']} | {_pct(row['coverage'])} | "
            f"{_pct(row['accuracy'])} | {_pct(row['accuracy_ci95']['low'])} | "
            f"{row['threshold']:.4f} |"
        )
    return "\n".join(lines)


def _analyze_key(
    examples: list[SelectiveExample],
    *,
    confidence_key: str,
    targets: list[float],
    min_accepted: int,
) -> dict[str, Any]:
    if confidence_key not in {"chosen_probability", "probability_margin", "score_margin"}:
        raise ValueError(f"unsupported confidence_key={confidence_key!r}")
    points = _prefix_points(examples, confidence_key=confidence_key)
    eligible = [row for row in points if row["accepted"] >= min_accepted]
    if not eligible:
        raise ValueError(
            f"min_accepted={min_accepted} exceeds available examples ({len(examples)})"
        )
    best_lower = max(
        eligible,
        key=lambda row: (
            row["accuracy_ci95"]["low"],
            row["accuracy"],
            row["accepted"],
            row["threshold"],
        ),
    )
    target_rows = [
        _recommend_for_target(eligible, target=target, best_lower=best_lower)
        for target in targets
    ]
    return {
        "confidence_key": confidence_key,
        "target_recommendations": target_rows,
        "best_lower_bound_prefix": best_lower,
        "prefix_curve": points,
    }


def _prefix_points(
    examples: list[SelectiveExample],
    *,
    confidence_key: str,
) -> list[dict[str, Any]]:
    ordered = sorted(
        examples,
        key=lambda item: (-item.confidence_value(confidence_key), item.problem_id),
    )
    n = len(ordered)
    correct = 0
    points = []
    for idx, item in enumerate(ordered, start=1):
        correct += int(item.selected_correct)
        prop = proportion_summary(correct, idx).to_json_dict()
        points.append(
            {
                "accepted": int(idx),
                "coverage": idx / float(n),
                "selected_correct": int(correct),
                "accuracy": float(prop["rate"]),
                "accuracy_ci95": {
                    "low": float(prop["ci_low"]),
                    "high": float(prop["ci_high"]),
                },
                "risk": 1.0 - float(prop["rate"]),
                "risk_ci95": {
                    "low": 1.0 - float(prop["ci_high"]),
                    "high": 1.0 - float(prop["ci_low"]),
                },
                "threshold": float(item.confidence_value(confidence_key)),
            }
        )
    return points


def _recommend_for_target(
    eligible: list[dict[str, Any]],
    *,
    target: float,
    best_lower: dict[str, Any],
) -> dict[str, Any]:
    certified = [
        row for row in eligible if float(row["accuracy_ci95"]["low"]) >= target
    ]
    if certified:
        chosen = max(
            certified,
            key=lambda row: (
                row["coverage"],
                row["accuracy_ci95"]["low"],
                row["accuracy"],
                row["threshold"],
            ),
        )
        return {**chosen, "target_accuracy": float(target), "meets_target": True}
    shortfall = max(0.0, float(target) - float(best_lower["accuracy_ci95"]["low"]))
    return {
        **best_lower,
        "target_accuracy": float(target),
        "meets_target": False,
        "lcb_shortfall": float(shortfall),
    }


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


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


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
        help="Comma-separated confidence keys to evaluate.",
    )
    parser.add_argument(
        "--target-accuracies",
        default=",".join(str(value) for value in DEFAULT_TARGET_ACCURACIES),
        help="Comma-separated target accuracies for Wilson-LCB certification.",
    )
    parser.add_argument("--min-accepted", type=int, default=5)
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
    summary = analyze_thresholds(
        examples,
        confidence_keys=_parse_keys(str(args.confidence_keys)),
        target_accuracies=_parse_floats(
            str(args.target_accuracies),
            default=DEFAULT_TARGET_ACCURACIES,
        ),
        min_accepted=int(args.min_accepted),
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
