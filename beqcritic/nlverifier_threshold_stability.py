"""Leave-one-out stability audit for NLVerifier abstention thresholds.

Threshold recommendations are useful only if small labeled-sample changes do
not make the deployed cutoff swing wildly. This module recomputes the selected
threshold after omitting each problem once, then reports how stable the cutoff
and accepted set are.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .nlverifier_selective import SelectiveExample, load_selective_examples
from .nlverifier_thresholds import analyze_thresholds
from .statistics import proportion_summary


SUPPORTED_CONFIDENCE_KEYS = {
    "chosen_probability",
    "probability_margin",
    "score_margin",
}


def analyze_threshold_stability(
    examples: Iterable[SelectiveExample],
    *,
    confidence_key: str = "chosen_probability",
    target_accuracy: float = 0.5,
    min_accepted: int = 5,
    top_cases: int = 15,
) -> dict[str, Any]:
    examples = sorted(list(examples), key=lambda item: item.problem_id)
    if not examples:
        raise ValueError("no selective examples found")
    if confidence_key not in SUPPORTED_CONFIDENCE_KEYS:
        raise ValueError(f"unsupported confidence_key={confidence_key!r}")
    min_accepted = max(1, int(min_accepted))
    if len(examples) - 1 < min_accepted:
        raise ValueError(
            f"min_accepted={min_accepted} leaves no valid leave-one-out folds "
            f"for {len(examples)} examples"
        )
    target_accuracy = min(1.0, max(0.0, float(target_accuracy)))

    full_row = _recommendation(
        examples,
        confidence_key=confidence_key,
        target_accuracy=target_accuracy,
        min_accepted=min_accepted,
    )
    full_threshold = float(full_row["threshold"])
    full_applied = _threshold_stats(
        examples,
        confidence_key=confidence_key,
        threshold=full_threshold,
    )
    full_ids = set(full_applied["accepted_problem_ids"])

    rows = []
    heldout_correct = 0
    heldout_accepted = 0
    for omitted in examples:
        train = [item for item in examples if item.problem_id != omitted.problem_id]
        row = _recommendation(
            train,
            confidence_key=confidence_key,
            target_accuracy=target_accuracy,
            min_accepted=min_accepted,
        )
        threshold = float(row["threshold"])
        train_stats = _threshold_stats(
            train,
            confidence_key=confidence_key,
            threshold=threshold,
        )
        full_stats = _threshold_stats(
            examples,
            confidence_key=confidence_key,
            threshold=threshold,
        )
        full_would_accept = omitted.problem_id in full_ids
        omitted_confidence = omitted.confidence_value(confidence_key)
        omitted_accepted = omitted_confidence >= threshold
        heldout_accepted += int(omitted_accepted)
        heldout_correct += int(omitted_accepted and omitted.selected_correct)
        rows.append(
            {
                "omitted_problem_id": omitted.problem_id,
                "omitted_selected_correct": bool(omitted.selected_correct),
                "omitted_has_any_correct": bool(omitted.has_any_correct),
                "omitted_confidence": float(omitted_confidence),
                "full_threshold_would_accept_omitted": bool(full_would_accept),
                "loo_threshold_would_accept_omitted": bool(omitted_accepted),
                "loo_threshold": threshold,
                "threshold_delta": threshold - full_threshold,
                "meets_target": bool(row.get("meets_target", False)),
                "train": _compact_stats(train_stats),
                "applied_to_full_sample": _compact_stats(full_stats),
                "accepted_delta_from_full": int(full_stats["accepted"])
                - int(full_applied["accepted"]),
                "accepted_set_jaccard_with_full": _jaccard(
                    full_ids,
                    set(full_stats["accepted_problem_ids"]),
                ),
            }
        )

    thresholds = [float(row["loo_threshold"]) for row in rows]
    changed = [
        row for row in rows if abs(float(row["threshold_delta"])) > 1e-12
    ]
    jaccards = [float(row["accepted_set_jaccard_with_full"]) for row in rows]
    all_accepted = [
        int(row["applied_to_full_sample"]["accepted"]) for row in rows
    ]
    train_accepted = [int(row["train"]["accepted"]) for row in rows]
    heldout = proportion_summary(heldout_correct, heldout_accepted).to_json_dict()
    threshold_counts = Counter(_threshold_bucket(value) for value in thresholds)
    sensitive = sorted(
        rows,
        key=lambda row: (
            abs(float(row["threshold_delta"])),
            abs(int(row["accepted_delta_from_full"])),
            row["omitted_problem_id"],
        ),
        reverse=True,
    )[: max(0, int(top_cases))]

    selected_correct = sum(1 for item in examples if item.selected_correct)
    any_correct = sum(1 for item in examples if item.has_any_correct)
    return {
        "dataset": {
            "problems": len(examples),
            "has_any_correct": int(any_correct),
            "has_any_correct_rate": any_correct / float(len(examples)),
            "selected_correct": int(selected_correct),
            "selected_accuracy": selected_correct / float(len(examples)),
        },
        "policy": {
            "confidence_key": confidence_key,
            "target_accuracy": target_accuracy,
            "min_accepted": min_accepted,
        },
        "full_recommendation": {
            **full_row,
            "applied_to_full_sample": _compact_stats(full_applied),
        },
        "leave_one_out": {
            "resamples": len(rows),
            "unique_threshold_count": len(threshold_counts),
            "threshold_changed": len(changed),
            "threshold_changed_rate": len(changed) / float(len(rows)),
            "threshold_min": min(thresholds),
            "threshold_max": max(thresholds),
            "threshold_mean": mean(thresholds),
            "train_accepted_min": min(train_accepted),
            "train_accepted_max": max(train_accepted),
            "applied_full_accepted_min": min(all_accepted),
            "applied_full_accepted_max": max(all_accepted),
            "accepted_set_jaccard_min": min(jaccards),
            "accepted_set_jaccard_mean": mean(jaccards),
            "meets_target": sum(1 for row in rows if row["meets_target"]),
            "heldout": {
                **heldout,
                "accepted": int(heldout_accepted),
                "rejected": int(len(rows) - heldout_accepted),
            },
        },
        "threshold_values": [
            {"threshold": float(value), "resamples": int(count)}
            for value, count in sorted(
                threshold_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "sensitive_omissions": sensitive,
        "leave_one_out_rows": rows,
    }


def format_markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    policy = summary["policy"]
    full = summary["full_recommendation"]
    full_stats = full["applied_to_full_sample"]
    loo = summary["leave_one_out"]
    heldout = loo["heldout"]
    lines = [
        "# NLVerifier Threshold Stability",
        "",
        f"- Problems: {dataset['problems']}",
        f"- Any-correct ceiling: {_pct(dataset['has_any_correct_rate'])} "
        f"({dataset['has_any_correct']}/{dataset['problems']})",
        f"- Full-coverage selected accuracy: {_pct(dataset['selected_accuracy'])} "
        f"({dataset['selected_correct']}/{dataset['problems']})",
        f"- Confidence key: `{policy['confidence_key']}`",
        f"- Target accuracy: {_pct(policy['target_accuracy'])}",
        f"- Minimum accepted examples: `{policy['min_accepted']}`",
        "",
        "## Full Recommendation",
        "",
        "| certified? | threshold | accepted | coverage | accuracy | 95% LCB |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {_yes_no(bool(full.get('meets_target')))} | {full['threshold']:.4f} | "
        f"{full_stats['accepted']} | {_pct(full_stats['coverage'])} | "
        f"{_pct(full_stats['accuracy']['rate'])} | "
        f"{_pct(full_stats['accuracy']['ci_low'])} |",
        "",
        "## Leave-One-Out Summary",
        "",
        f"- Resamples: {loo['resamples']}",
        f"- Unique thresholds: {loo['unique_threshold_count']}",
        f"- Threshold changed vs full: {_count_pct(loo['threshold_changed'], loo['resamples'])}",
        f"- Threshold range: `{loo['threshold_min']:.4f}` to `{loo['threshold_max']:.4f}`",
        f"- Train accepted range: {loo['train_accepted_min']} to {loo['train_accepted_max']}",
        f"- Applied-to-full accepted range: {loo['applied_full_accepted_min']} to {loo['applied_full_accepted_max']}",
        f"- Minimum accepted-set Jaccard vs full: {_pct(loo['accepted_set_jaccard_min'])}",
        f"- Held-out accepted accuracy: {_pct(heldout['rate'])} "
        f"({heldout['successes']}/{heldout['total']}; "
        f"accepted {heldout['accepted']}, rejected {heldout['rejected']})",
        "",
        "## Threshold Values",
        "",
        "| threshold | resamples |",
        "|---:|---:|",
    ]
    for row in summary["threshold_values"]:
        lines.append(f"| {row['threshold']:.4f} | {row['resamples']} |")

    lines.extend(
        [
            "",
            "## Most Sensitive Omissions",
            "",
            "| omitted problem | correct? | confidence | LOO threshold | delta | full accepted | Jaccard | held out accepted? |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["sensitive_omissions"]:
        lines.append(
            f"| {_md(row['omitted_problem_id'])} | "
            f"{_yes_no(bool(row['omitted_selected_correct']))} | "
            f"{row['omitted_confidence']:.4f} | "
            f"{row['loo_threshold']:.4f} | "
            f"{row['threshold_delta']:+.4f} | "
            f"{row['applied_to_full_sample']['accepted']} | "
            f"{_pct(row['accepted_set_jaccard_with_full'])} | "
            f"{_yes_no(bool(row['loo_threshold_would_accept_omitted']))} |"
        )
    return "\n".join(lines)


def _recommendation(
    examples: list[SelectiveExample],
    *,
    confidence_key: str,
    target_accuracy: float,
    min_accepted: int,
) -> dict[str, Any]:
    summary = analyze_thresholds(
        examples,
        confidence_keys=[confidence_key],
        target_accuracies=[target_accuracy],
        min_accepted=min_accepted,
    )
    return dict(summary["signals"][confidence_key]["target_recommendations"][0])


def _threshold_stats(
    examples: list[SelectiveExample],
    *,
    confidence_key: str,
    threshold: float,
) -> dict[str, Any]:
    accepted = [
        item
        for item in sorted(examples, key=lambda row: row.problem_id)
        if item.confidence_value(confidence_key) >= threshold
    ]
    selected_correct = sum(1 for item in accepted if item.selected_correct)
    any_correct = sum(1 for item in accepted if item.has_any_correct)
    total = len(examples)
    return {
        "accepted": int(len(accepted)),
        "coverage": 0.0 if total == 0 else len(accepted) / float(total),
        "selected_correct": int(selected_correct),
        "has_any_correct": int(any_correct),
        "accuracy": proportion_summary(selected_correct, len(accepted)).to_json_dict(),
        "oracle_ceiling": proportion_summary(any_correct, len(accepted)).to_json_dict(),
        "accepted_problem_ids": [item.problem_id for item in accepted],
    }


def _compact_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": int(stats["accepted"]),
        "coverage": float(stats["coverage"]),
        "selected_correct": int(stats["selected_correct"]),
        "has_any_correct": int(stats["has_any_correct"]),
        "accuracy": stats["accuracy"],
        "oracle_ceiling": stats["oracle_ceiling"],
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / float(len(left | right))


def _threshold_bucket(value: float) -> float:
    return round(float(value), 12)


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _count_pct(count: int, total: int) -> str:
    return f"{count}/{total} ({_pct(0.0 if total == 0 else count / float(total))})"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _md(value: object) -> str:
    return str(value).replace("|", "\\|")


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
        "--confidence-key",
        default="chosen_probability",
        choices=tuple(sorted(SUPPORTED_CONFIDENCE_KEYS)),
    )
    parser.add_argument("--target-accuracy", type=float, default=0.5)
    parser.add_argument("--min-accepted", type=int, default=5)
    parser.add_argument("--top-cases", type=int, default=15)
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
    summary = analyze_threshold_stability(
        examples,
        confidence_key=str(args.confidence_key),
        target_accuracy=float(args.target_accuracy),
        min_accepted=int(args.min_accepted),
        top_cases=int(args.top_cases),
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
