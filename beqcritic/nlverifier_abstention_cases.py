"""Build an inspectable casebook for NLVerifier abstention decisions.

The abstention policy summary gives aggregate coverage and accuracy. This
module joins the accepted/abstained rows back to the scored candidate groups so
that accepted errors, abstained correct selections, and abstained missed
opportunities can be inspected with candidate text, ranks, scores, and margins.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def analyze_cases(
    *,
    scores_path: str | Path,
    accepted_path: str | Path,
    abstained_path: str | Path,
    score_key: str = "scores",
    top_candidates: int = 5,
    max_cases: int = 20,
) -> dict[str, Any]:
    source_rows = _load_jsonl_map(scores_path)
    accepted_rows = _load_jsonl_rows(accepted_path)
    abstained_rows = _load_jsonl_rows(abstained_path)
    if not accepted_rows and not abstained_rows:
        raise ValueError("no accepted or abstained rows found")

    cases: list[dict[str, Any]] = []
    for row in accepted_rows:
        cases.append(
            _build_case(
                source_rows,
                row,
                bucket="accepted",
                score_key=score_key,
                top_candidates=top_candidates,
            )
        )
    for row in abstained_rows:
        cases.append(
            _build_case(
                source_rows,
                row,
                bucket="abstained",
                score_key=score_key,
                top_candidates=top_candidates,
            )
        )

    counts = Counter(str(case["case_kind"]) for case in cases)
    accepted = [case for case in cases if case["bucket"] == "accepted"]
    abstained = [case for case in cases if case["bucket"] == "abstained"]
    threshold_values = sorted(
        {
            float(case["threshold"])
            for case in cases
            if case.get("threshold") is not None
        }
    )
    confidence_keys = sorted(
        {
            str(case["confidence_key"])
            for case in cases
            if case.get("confidence_key")
        }
    )

    return {
        "dataset": {
            "problems": int(len(cases)),
            "accepted": int(len(accepted)),
            "abstained": int(len(abstained)),
            "accepted_correct": int(sum(case["selected_correct"] for case in accepted)),
            "accepted_errors": int(sum(not case["selected_correct"] for case in accepted)),
            "accepted_missed_available_correct": int(counts["accepted_missed_available_correct"]),
            "accepted_no_correct_candidate": int(counts["accepted_no_correct_candidate"]),
            "abstained_selected_correct": int(counts["abstained_correct_selection"]),
            "abstained_missed_available_correct": int(counts["abstained_missed_available_correct"]),
            "abstained_no_correct_candidate": int(counts["abstained_no_correct_candidate"]),
        },
        "threshold": {
            "confidence_keys": confidence_keys,
            "thresholds": threshold_values,
        },
        "case_counts": dict(sorted(counts.items())),
        "casebook": {
            "accepted_errors": _top_cases(
                (case for case in accepted if not case["selected_correct"]),
                max_cases=max_cases,
                sort_by_confidence_desc=True,
            ),
            "accepted_missed_available_correct": _top_cases(
                (case for case in accepted if case["case_kind"] == "accepted_missed_available_correct"),
                max_cases=max_cases,
                sort_by_confidence_desc=True,
            ),
            "abstained_correct_selections": _top_cases(
                (case for case in abstained if case["case_kind"] == "abstained_correct_selection"),
                max_cases=max_cases,
                sort_by_confidence_desc=True,
            ),
            "abstained_missed_available_correct": _top_cases(
                (case for case in abstained if case["case_kind"] == "abstained_missed_available_correct"),
                max_cases=max_cases,
                sort_by_confidence_desc=True,
            ),
            "highest_confidence_abstentions": _top_cases(
                abstained,
                max_cases=max_cases,
                sort_by_confidence_desc=True,
            ),
        },
        "cases": cases,
        "config": {
            "score_key": str(score_key),
            "top_candidates": int(max(1, top_candidates)),
            "max_cases": int(max(0, max_cases)),
        },
    }


def format_markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    threshold = summary["threshold"]
    casebook = summary["casebook"]
    lines = [
        "# NLVerifier Abstention Casebook",
        "",
        f"- Problems: {dataset['problems']}",
        f"- Accepted: {dataset['accepted']}",
        f"- Abstained: {dataset['abstained']}",
        f"- Accepted correct: {dataset['accepted_correct']}",
        f"- Accepted errors: {dataset['accepted_errors']}",
        f"- Accepted misses with a correct candidate available: {dataset['accepted_missed_available_correct']}",
        f"- Abstained correct selections: {dataset['abstained_selected_correct']}",
        f"- Abstained misses with a correct candidate available: {dataset['abstained_missed_available_correct']}",
        f"- Confidence keys: `{', '.join(threshold['confidence_keys'])}`",
        f"- Thresholds: `{', '.join(f'{value:.4f}' for value in threshold['thresholds'])}`",
        "",
        "## Case Counts",
        "",
        "| case kind | count |",
        "|---|---:|",
    ]
    for name, count in summary["case_counts"].items():
        lines.append(f"| `{_md(name)}` | {count} |")

    _append_case_table(
        lines,
        "Accepted Errors",
        casebook["accepted_errors"],
        include_best=True,
    )
    _append_case_table(
        lines,
        "Accepted Misses With Correct Candidate Available",
        casebook["accepted_missed_available_correct"],
        include_best=True,
    )
    _append_case_table(
        lines,
        "Abstained Correct Selections",
        casebook["abstained_correct_selections"],
        include_best=False,
    )
    _append_case_table(
        lines,
        "Abstained Misses With Correct Candidate Available",
        casebook["abstained_missed_available_correct"],
        include_best=True,
    )
    _append_case_table(
        lines,
        "Highest-Confidence Abstentions",
        casebook["highest_confidence_abstentions"],
        include_best=True,
    )
    return "\n".join(lines).rstrip()


def write_cases_jsonl(path: str | Path, cases: Iterable[dict[str, Any]]) -> None:
    Path(path).write_text(
        "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )


def _build_case(
    source_rows: dict[str, dict[str, Any]],
    row: dict[str, Any],
    *,
    bucket: str,
    score_key: str,
    top_candidates: int,
) -> dict[str, Any]:
    problem_id = str(row.get("problem_id", ""))
    if problem_id not in source_rows:
        raise ValueError(f"problem_id={problem_id!r} not found in scores file")
    source = source_rows[problem_id]
    candidates = source.get("candidates") or []
    labels = source.get("labels") or []
    scores = source.get(score_key)
    if not isinstance(candidates, list) or not isinstance(labels, list):
        raise ValueError(f"candidates/labels must be lists for problem_id={problem_id!r}")
    if len(candidates) != len(labels):
        raise ValueError(
            f"candidates/labels length mismatch for {problem_id}: {len(candidates)} vs {len(labels)}"
        )
    if not isinstance(scores, list):
        raise ValueError(f"{score_key!r} must be a list for problem_id={problem_id!r}")
    if len(scores) != len(candidates):
        raise ValueError(
            f"score length mismatch for {problem_id}: {len(scores)} vs {len(candidates)}"
        )
    if not candidates:
        raise ValueError(f"empty candidate set for problem_id={problem_id!r}")

    labels01 = [1 if int(value) else 0 for value in labels]
    scores_f = [float(value) for value in scores]
    chosen_index = int(row.get("chosen_index"))
    if chosen_index < 0 or chosen_index >= len(candidates):
        raise ValueError(
            f"chosen_index out of range for problem_id={problem_id!r}: {chosen_index}"
        )
    order = sorted(range(len(scores_f)), key=lambda i: (-scores_f[i], i))
    rank_by_index = {idx: rank for rank, idx in enumerate(order, start=1)}
    correct_indices = [idx for idx, label in enumerate(labels01) if label]
    best_correct_index = (
        max(correct_indices, key=lambda idx: (scores_f[idx], -idx))
        if correct_indices
        else None
    )
    selected_correct = bool(labels01[chosen_index])
    has_any_correct = bool(correct_indices)
    confidence = _optional_float(row.get("confidence", row.get("chosen_probability")))
    threshold = _optional_float(row.get("threshold"))
    best_correct_score = scores_f[best_correct_index] if best_correct_index is not None else None
    chosen_score = scores_f[chosen_index]
    case_kind = _case_kind(
        bucket=bucket,
        selected_correct=selected_correct,
        has_any_correct=has_any_correct,
    )

    return {
        "problem_id": problem_id,
        "bucket": bucket,
        "case_kind": case_kind,
        "confidence_key": row.get("confidence_key"),
        "confidence": confidence,
        "threshold": threshold,
        "threshold_distance": (
            None if confidence is None or threshold is None else confidence - threshold
        ),
        "selected_correct": selected_correct,
        "has_any_correct": has_any_correct,
        "chosen_index": chosen_index,
        "chosen_rank": int(rank_by_index[chosen_index]),
        "chosen_score": chosen_score,
        "chosen_candidate": str(candidates[chosen_index]),
        "best_correct_index": best_correct_index,
        "best_correct_rank": (
            int(rank_by_index[best_correct_index]) if best_correct_index is not None else None
        ),
        "best_correct_score": best_correct_score,
        "best_correct_candidate": (
            str(candidates[best_correct_index]) if best_correct_index is not None else None
        ),
        "score_gap_to_best_correct": (
            None if best_correct_score is None else chosen_score - best_correct_score
        ),
        "n_candidates": int(len(candidates)),
        "top_candidates": [
            {
                "rank": int(rank),
                "index": int(idx),
                "score": float(scores_f[idx]),
                "label": int(labels01[idx]),
                "is_chosen": bool(idx == chosen_index),
                "candidate": str(candidates[idx]),
            }
            for rank, idx in enumerate(order[: max(1, int(top_candidates))], start=1)
        ],
    }


def _case_kind(*, bucket: str, selected_correct: bool, has_any_correct: bool) -> str:
    if bucket == "accepted":
        if selected_correct:
            return "accepted_correct"
        if has_any_correct:
            return "accepted_missed_available_correct"
        return "accepted_no_correct_candidate"
    if selected_correct:
        return "abstained_correct_selection"
    if has_any_correct:
        return "abstained_missed_available_correct"
    return "abstained_no_correct_candidate"


def _top_cases(
    cases: Iterable[dict[str, Any]],
    *,
    max_cases: int,
    sort_by_confidence_desc: bool,
) -> list[dict[str, Any]]:
    rows = list(cases)
    rows.sort(
        key=lambda case: (
            -float(case["confidence"] or 0.0) if sort_by_confidence_desc else 0.0,
            str(case["problem_id"]),
        )
    )
    return rows[: max(0, int(max_cases))]


def _append_case_table(
    lines: list[str],
    title: str,
    cases: list[dict[str, Any]],
    *,
    include_best: bool,
) -> None:
    lines.extend(["", f"## {title}", ""])
    if not cases:
        lines.append("No cases.")
        return
    if include_best:
        lines.extend(
            [
                "| problem_id | bucket | confidence | threshold delta | chosen | chosen rank | best correct | best rank | score gap |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for case in cases:
            lines.append(
                f"| {_md(case['problem_id'])} | {case['bucket']} | {_fmt(case['confidence'])} | "
                f"{_fmt(case['threshold_distance'])} | {case['chosen_index']} | "
                f"{case['chosen_rank']} | {_fmt_index(case['best_correct_index'])} | "
                f"{_fmt_index(case['best_correct_rank'])} | {_fmt(case['score_gap_to_best_correct'])} |"
            )
        return

    lines.extend(
        [
            "| problem_id | confidence | threshold delta | chosen | chosen rank | score | n candidates |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in cases:
        lines.append(
            f"| {_md(case['problem_id'])} | {_fmt(case['confidence'])} | "
            f"{_fmt(case['threshold_distance'])} | {case['chosen_index']} | "
            f"{case['chosen_rank']} | {_fmt(case['chosen_score'])} | {case['n_candidates']} |"
        )


def _load_jsonl_map(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = _load_jsonl_rows(path)
    return {str(row["problem_id"]): row for row in rows}


def _load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("problem_id") is None:
                raise ValueError(f"missing problem_id at {path}:{line_no}")
            rows.append(row)
    return rows


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _fmt_index(value: Any) -> str:
    if value is None:
        return "-"
    return str(int(value))


def _md(value: object) -> str:
    return str(value).replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--abstained", required=True)
    parser.add_argument("--score-key", default="scores")
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    parser.add_argument("--cases-jsonl")
    args = parser.parse_args()

    summary = analyze_cases(
        scores_path=args.scores,
        accepted_path=args.accepted,
        abstained_path=args.abstained,
        score_key=str(args.score_key),
        top_candidates=int(args.top_candidates),
        max_cases=int(args.max_cases),
    )
    markdown = format_markdown(summary)
    print(markdown)

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_md:
        Path(args.output_md).write_text(markdown + "\n", encoding="utf-8")
    if args.cases_jsonl:
        write_cases_jsonl(args.cases_jsonl, summary["cases"])


if __name__ == "__main__":
    main()
