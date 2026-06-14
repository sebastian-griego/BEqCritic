"""Apply NLVerifier abstention thresholds to scored candidate groups.

The threshold analyzers recommend confidence cutoffs. This module applies a
chosen cutoff to produce an accepted-selection JSONL, an abstention JSONL, and
an audit report with coverage and accepted-accuracy confidence intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .nlverifier_selective import SelectiveExample, load_selective_examples
from .statistics import proportion_summary


SUPPORTED_CONFIDENCE_KEYS = {
    "chosen_probability",
    "probability_margin",
    "score_margin",
}


def apply_abstention(
    examples: list[SelectiveExample],
    source_rows: dict[str, dict[str, Any]],
    *,
    confidence_key: str = "chosen_probability",
    threshold: float,
    max_examples: int = 15,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not examples:
        raise ValueError("no selective examples found")
    _validate_confidence_key(confidence_key)
    threshold = float(threshold)

    accepted: list[dict[str, Any]] = []
    abstained: list[dict[str, Any]] = []
    annotated: list[dict[str, Any]] = []
    for example in sorted(examples, key=lambda item: item.problem_id):
        confidence = example.confidence_value(confidence_key)
        row = _selection_row(
            example,
            source_rows.get(example.problem_id, {}),
            confidence_key=confidence_key,
            confidence=confidence,
            threshold=threshold,
        )
        annotated.append(row)
        if confidence >= threshold:
            accepted.append(row)
        else:
            abstained.append(row)

    report = _report(
        annotated,
        accepted,
        abstained,
        confidence_key=confidence_key,
        threshold=threshold,
        max_examples=max_examples,
    )
    return accepted, abstained, report


def resolve_threshold(
    *,
    threshold: float | None = None,
    thresholds_json: str | Path | None = None,
    confidence_key: str = "chosen_probability",
    target_accuracy: float | None = None,
    require_certified: bool = False,
) -> dict[str, Any]:
    _validate_confidence_key(confidence_key)
    if threshold is not None:
        return {
            "confidence_key": confidence_key,
            "threshold": float(threshold),
            "source": "explicit",
            "target_accuracy": target_accuracy,
            "meets_target": None,
        }
    if thresholds_json is None or not str(thresholds_json).strip():
        raise ValueError("provide either threshold or thresholds_json")

    payload = json.loads(Path(thresholds_json).read_text(encoding="utf-8"))
    signal = payload.get("signals", {}).get(confidence_key)
    if not isinstance(signal, dict):
        raise ValueError(f"confidence key {confidence_key!r} not found in thresholds JSON")

    if target_accuracy is None:
        row = signal.get("best_lower_bound_prefix")
        source = "best_lower_bound_prefix"
    else:
        row = _find_target_row(signal, float(target_accuracy))
        source = "target_recommendation"
    if not isinstance(row, dict):
        raise ValueError("threshold row not found")
    if require_certified and row.get("meets_target") is False:
        raise ValueError(
            f"threshold for {confidence_key!r} does not certify target_accuracy={target_accuracy}"
        )
    return {
        "confidence_key": confidence_key,
        "threshold": float(row["threshold"]),
        "source": source,
        "target_accuracy": (
            float(row["target_accuracy"])
            if "target_accuracy" in row
            else target_accuracy
        ),
        "meets_target": row.get("meets_target"),
        "accepted": int(row["accepted"]) if "accepted" in row else None,
        "coverage": float(row["coverage"]) if "coverage" in row else None,
        "accuracy": float(row["accuracy"]) if "accuracy" in row else None,
        "accuracy_ci95": row.get("accuracy_ci95"),
    }


def load_source_rows(
    *,
    scored_path: str | Path | None = None,
    candidates_path: str | Path | None = None,
    selections_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    if scored_path is not None:
        return _load_jsonl_map(scored_path)
    if candidates_path is None or selections_path is None:
        raise ValueError("provide either scored_path or both candidates_path and selections_path")

    candidates = _load_jsonl_map(candidates_path)
    selections = _load_jsonl_map(selections_path)
    common = sorted(set(candidates) & set(selections))
    if not common:
        raise ValueError("no overlapping problem_ids across candidates and selections")
    rows: dict[str, dict[str, Any]] = {}
    for problem_id in common:
        merged = dict(candidates[problem_id])
        merged.update(selections[problem_id])
        rows[problem_id] = merged
    return rows


def format_markdown(report: dict[str, Any]) -> str:
    threshold = report["threshold"]
    dataset = report["dataset"]
    accepted_accuracy = dataset["accepted_accuracy"]
    lines = [
        "# NLVerifier Abstention Policy",
        "",
        f"- Confidence key: `{threshold['confidence_key']}`",
        f"- Threshold: `{threshold['threshold']:.4f}`",
        f"- Threshold source: `{threshold['source']}`",
    ]
    if threshold.get("target_accuracy") is not None:
        lines.append(f"- Target accuracy: {_pct(threshold['target_accuracy'])}")
    if threshold.get("meets_target") is not None:
        lines.append(f"- Wilson-LCB certified target: {_yes_no(bool(threshold['meets_target']))}")
    lines.extend(
        [
            f"- Problems: {dataset['problems']}",
            f"- Accepted: {dataset['accepted']} ({_pct(dataset['coverage'])})",
            f"- Abstained: {dataset['abstained']} ({_pct(dataset['abstention_rate'])})",
            f"- Full-coverage accuracy: {_pct(dataset['full_accuracy']['rate'])} "
            f"({dataset['full_accuracy']['successes']}/{dataset['full_accuracy']['total']})",
            f"- Accepted accuracy: {_pct(accepted_accuracy['rate'])} "
            f"[{_pct(accepted_accuracy['ci_low'])}, {_pct(accepted_accuracy['ci_high'])}] "
            f"({accepted_accuracy['successes']}/{accepted_accuracy['total']})",
            "",
            "## Outcome Counts",
            "",
            "| bucket | count | selected correct | accuracy |",
            "|---|---:|---:|---:|",
            f"| accepted | {dataset['accepted']} | {dataset['accepted_correct']} | {_pct(dataset['accepted_accuracy']['rate'])} |",
            f"| abstained | {dataset['abstained']} | {dataset['abstained_correct']} | {_pct(dataset['abstained_accuracy']['rate'])} |",
            "",
            "## Accepted Errors",
            "",
        ]
    )
    accepted_errors = report.get("accepted_errors", [])
    if accepted_errors:
        lines.extend(
            [
                "| problem_id | confidence | chosen |",
                "|---|---:|---:|",
            ]
        )
        for row in accepted_errors:
            lines.append(
                f"| {_md(row['problem_id'])} | {row['confidence']:.4f} | {row['chosen_index']} |"
            )
    else:
        lines.append("No accepted errors.")

    lines.extend(["", "## Highest-Confidence Abstentions", ""])
    abstentions = report.get("highest_confidence_abstentions", [])
    if abstentions:
        lines.extend(
            [
                "| problem_id | confidence | selected correct | chosen |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in abstentions:
            lines.append(
                f"| {_md(row['problem_id'])} | {row['confidence']:.4f} | "
                f"{_yes_no(row['selected_correct'])} | {row['chosen_index']} |"
            )
    else:
        lines.append("No abstentions.")
    return "\n".join(lines)


def _selection_row(
    example: SelectiveExample,
    source_row: dict[str, Any],
    *,
    confidence_key: str,
    confidence: float,
    threshold: float,
) -> dict[str, Any]:
    candidates = source_row.get("candidates")
    chosen = None
    if isinstance(candidates, list) and 0 <= example.chosen_index < len(candidates):
        chosen = candidates[example.chosen_index]
    return {
        "problem_id": example.problem_id,
        "chosen_index": int(example.chosen_index),
        "chosen": chosen,
        "score": float(example.chosen_score),
        "accepted": bool(confidence >= threshold),
        "abstained": bool(confidence < threshold),
        "confidence_key": confidence_key,
        "confidence": float(confidence),
        "threshold": float(threshold),
        "chosen_probability": float(example.chosen_probability),
        "probability_margin": _optional_float(example.probability_margin),
        "score_margin": _optional_float(example.score_margin),
        "runner_up_index": example.runner_up_index,
        "runner_up_score": _optional_float(example.runner_up_score),
        "runner_up_probability": _optional_float(example.runner_up_probability),
        "n_candidates": int(example.n_candidates),
        "selected_correct": bool(example.selected_correct),
        "has_any_correct": bool(example.has_any_correct),
    }


def _report(
    annotated: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    abstained: list[dict[str, Any]],
    *,
    confidence_key: str,
    threshold: float,
    max_examples: int,
) -> dict[str, Any]:
    total = len(annotated)
    accepted_correct = sum(1 for row in accepted if row["selected_correct"])
    abstained_correct = sum(1 for row in abstained if row["selected_correct"])
    full_correct = accepted_correct + abstained_correct
    any_correct = sum(1 for row in annotated if row["has_any_correct"])
    accepted_errors = sorted(
        (row for row in accepted if not row["selected_correct"]),
        key=lambda row: (-float(row["confidence"]), str(row["problem_id"])),
    )[: max(0, int(max_examples))]
    highest_confidence_abstentions = sorted(
        abstained,
        key=lambda row: (-float(row["confidence"]), str(row["problem_id"])),
    )[: max(0, int(max_examples))]
    return {
        "threshold": {
            "confidence_key": confidence_key,
            "threshold": float(threshold),
            "source": "applied",
        },
        "dataset": {
            "problems": int(total),
            "accepted": int(len(accepted)),
            "abstained": int(len(abstained)),
            "coverage": _rate(len(accepted), total),
            "abstention_rate": _rate(len(abstained), total),
            "has_any_correct": int(any_correct),
            "has_any_correct_rate": _rate(any_correct, total),
            "full_selected_correct": int(full_correct),
            "full_accuracy": proportion_summary(full_correct, total).to_json_dict(),
            "accepted_correct": int(accepted_correct),
            "accepted_accuracy": proportion_summary(accepted_correct, len(accepted)).to_json_dict(),
            "abstained_correct": int(abstained_correct),
            "abstained_accuracy": proportion_summary(abstained_correct, len(abstained)).to_json_dict(),
        },
        "accepted_errors": [_compact_example(row) for row in accepted_errors],
        "highest_confidence_abstentions": [
            _compact_example(row) for row in highest_confidence_abstentions
        ],
    }


def _compact_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem_id": row["problem_id"],
        "chosen_index": row["chosen_index"],
        "confidence": row["confidence"],
        "selected_correct": row["selected_correct"],
        "has_any_correct": row["has_any_correct"],
        "chosen_probability": row["chosen_probability"],
        "probability_margin": row["probability_margin"],
        "score_margin": row["score_margin"],
        "n_candidates": row["n_candidates"],
    }


def _find_target_row(signal: dict[str, Any], target_accuracy: float) -> dict[str, Any]:
    rows = signal.get("target_recommendations", [])
    if not isinstance(rows, list):
        raise ValueError("target_recommendations must be a list")
    for row in rows:
        if abs(float(row.get("target_accuracy", -1.0)) - target_accuracy) <= 1e-9:
            return row
    available = ", ".join(str(row.get("target_accuracy")) for row in rows if isinstance(row, dict))
    raise ValueError(f"target_accuracy={target_accuracy} not found; available: {available}")


def _load_jsonl_map(path: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = row.get("problem_id")
            if problem_id is None:
                raise ValueError(f"missing problem_id in {path}: {row}")
            out[str(problem_id)] = row
    return out


def _validate_confidence_key(confidence_key: str) -> None:
    if confidence_key not in SUPPORTED_CONFIDENCE_KEYS:
        keys = ", ".join(sorted(SUPPORTED_CONFIDENCE_KEYS))
        raise ValueError(f"unsupported confidence_key={confidence_key!r}; expected one of {keys}")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _rate(numer: int, denom: int) -> float:
    return (float(numer) / float(denom)) if denom else 0.0


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


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
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--thresholds-json", default="")
    parser.add_argument("--target-accuracy", type=float)
    parser.add_argument("--require-certified", action="store_true")
    parser.add_argument("--max-examples", type=int, default=15)
    parser.add_argument("--output-accepted", required=True)
    parser.add_argument("--output-abstained", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    resolved = resolve_threshold(
        threshold=args.threshold,
        thresholds_json=str(args.thresholds_json).strip() or None,
        confidence_key=str(args.confidence_key),
        target_accuracy=args.target_accuracy,
        require_certified=bool(args.require_certified),
    )
    examples = load_selective_examples(
        scored_path=args.scores,
        candidates_path=args.candidates,
        selections_path=args.selections,
        calibration_json=str(args.calibration_json).strip() or None,
        temperature=float(args.temperature),
        score_key=str(args.score_key),
        minimize=bool(args.minimize),
    )
    source_rows = load_source_rows(
        scored_path=args.scores,
        candidates_path=args.candidates,
        selections_path=args.selections,
    )
    accepted, abstained, report = apply_abstention(
        examples,
        source_rows,
        confidence_key=str(args.confidence_key),
        threshold=float(resolved["threshold"]),
        max_examples=int(args.max_examples),
    )
    report["threshold"].update(resolved)
    if args.calibration_json:
        report["threshold"]["calibration_json"] = str(args.calibration_json)
    if args.thresholds_json:
        report["threshold"]["thresholds_json"] = str(args.thresholds_json)
    markdown = format_markdown(report)

    Path(args.output_accepted).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted),
        encoding="utf-8",
    )
    Path(args.output_abstained).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in abstained),
        encoding="utf-8",
    )
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_md:
        Path(args.output_md).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
