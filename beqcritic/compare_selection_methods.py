"""Compare two selection JSONLs with confidence intervals and a paired test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .schema import validate_grouped_candidates
from .statistics import paired_comparison, proportion_summary


def _load_jsonl_map(path: str | Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            problem_id = record.get("problem_id")
            if problem_id is None:
                raise ValueError(f"missing problem_id at {path}:{line_no}")
            records[str(problem_id)] = record
    return records


def _chosen_indices(record: dict[str, Any]) -> list[int]:
    raw = record.get("chosen_indices")
    if isinstance(raw, list) and raw:
        return [int(value) for value in raw]
    if "chosen_index" in record:
        return [int(record["chosen_index"])]
    raise ValueError(f"selection record has no chosen_index/chosen_indices: {record!r}")


def _selected_correct(
    selection: dict[str, Any], labels: list[int], problem_id: str, method_name: str
) -> bool:
    chosen = _chosen_indices(selection)[0]
    if chosen < 0 or chosen >= len(labels):
        raise ValueError(
            f"{method_name} chosen index out of range for {problem_id}: {chosen} "
            f"(n={len(labels)})"
        )
    return bool(labels[chosen])


def compare_files(
    *,
    candidates_path: str | Path,
    selections_a_path: str | Path,
    selections_b_path: str | Path,
    a_name: str = "A",
    b_name: str = "B",
) -> dict[str, Any]:
    candidates = _load_jsonl_map(candidates_path)
    selections_a = _load_jsonl_map(selections_a_path)
    selections_b = _load_jsonl_map(selections_b_path)

    problem_ids = sorted(set(candidates) & set(selections_a) & set(selections_b))
    if not problem_ids:
        raise ValueError(
            "no overlapping problem_ids across candidates and both selection files"
        )

    a_success: list[bool] = []
    b_success: list[bool] = []
    any_correct: list[bool] = []
    total_candidates = 0
    total_candidate_pairs = 0

    for problem_id in problem_ids:
        grouped = validate_grouped_candidates(
            candidates[problem_id], require_labels=True
        )
        assert grouped.labels is not None
        if not grouped.candidates:
            continue
        labels = grouped.labels
        total_candidates += len(grouped.candidates)
        total_candidate_pairs += (
            len(grouped.candidates) * (len(grouped.candidates) - 1) // 2
        )
        any_correct.append(any(bool(label) for label in labels))
        a_success.append(
            _selected_correct(selections_a[problem_id], labels, problem_id, a_name)
        )
        b_success.append(
            _selected_correct(selections_b[problem_id], labels, problem_id, b_name)
        )

    if not a_success:
        raise ValueError("no non-empty candidate sets found")

    comparison = paired_comparison(a_success, b_success)
    any_total = sum(1 for value in any_correct if value)
    a_given_any = sum(
        1 for ok, reachable in zip(a_success, any_correct) if ok and reachable
    )
    b_given_any = sum(
        1 for ok, reachable in zip(b_success, any_correct) if ok and reachable
    )

    return {
        "dataset": {
            "problems": len(a_success),
            "has_any_correct": proportion_summary(
                any_total, len(any_correct)
            ).to_json_dict(),
            "avg_candidates_per_problem": total_candidates / float(len(a_success)),
            "total_candidate_pairs": total_candidate_pairs,
        },
        "a": {
            "name": a_name,
            "selected_correct": proportion_summary(
                comparison.a_successes,
                comparison.total,
            ).to_json_dict(),
            "selected_correct_given_any": proportion_summary(
                a_given_any, any_total
            ).to_json_dict(),
        },
        "b": {
            "name": b_name,
            "selected_correct": proportion_summary(
                comparison.b_successes,
                comparison.total,
            ).to_json_dict(),
            "selected_correct_given_any": proportion_summary(
                b_given_any, any_total
            ).to_json_dict(),
        },
        "paired": comparison.to_json_dict(),
    }


def format_markdown(summary: dict[str, Any]) -> str:
    a = summary["a"]
    b = summary["b"]
    paired = summary["paired"]
    dataset = summary["dataset"]
    lines = [
        f"# Selection comparison: {a['name']} vs {b['name']}",
        "",
        f"- Problems: {dataset['problems']}",
        "- Any-correct ceiling: "
        f"{_pct(dataset['has_any_correct']['rate'])} "
        f"({dataset['has_any_correct']['successes']}/{dataset['has_any_correct']['total']})",
        f"- Avg candidates/problem: {dataset['avg_candidates_per_problem']:.2f}",
        f"- Total candidate pairs: {dataset['total_candidate_pairs']}",
        "",
        "| method | selected correct | 95% CI | selected correct given any |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in (a, b):
        selected = method["selected_correct"]
        selected_any = method["selected_correct_given_any"]
        lines.append(
            f"| {method['name']} | "
            f"{_pct(selected['rate'])} ({selected['successes']}/{selected['total']}) | "
            f"[{_pct(selected['ci_low'])}, {_pct(selected['ci_high'])}] | "
            f"{_pct(selected_any['rate'])} ({selected_any['successes']}/{selected_any['total']}) |"
        )
    lines.extend(
        [
            "",
            "## Paired outcome",
            "",
            f"- Both correct: {paired['both_success']}",
            f"- {a['name']} only: {paired['a_only']}",
            f"- {b['name']} only: {paired['b_only']}",
            f"- Neither correct: {paired['neither_success']}",
            f"- {b['name']} - {a['name']}: {_pct(paired['b_minus_a'])}",
            f"- Discordant pairs: {paired['discordant']}",
            f"- Exact two-sided sign-test p-value: {paired['exact_sign_p']:.6g}",
            "",
        ]
    )
    return "\n".join(lines)


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--selections-a", required=True)
    parser.add_argument("--selections-b", required=True)
    parser.add_argument("--a-name", default="A")
    parser.add_argument("--b-name", default="B")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    summary = compare_files(
        candidates_path=args.candidates,
        selections_a_path=args.selections_a,
        selections_b_path=args.selections_b,
        a_name=args.a_name,
        b_name=args.b_name,
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
