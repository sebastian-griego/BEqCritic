"""Multi-method selection leaderboard with paired significance tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

from .jsonl import load_jsonl_map_by_problem_id, matching_problem_ids_many
from .schema import validate_grouped_candidates
from .statistics import paired_comparison, proportion_summary
from .textnorm import normalize_lean_statement


DERIVED_BASELINES = {"first", "shortest"}


def analyze_leaderboard(
    *,
    candidates_path: str | Path,
    selections: dict[str, str | Path],
    abstentions: dict[str, str | Path] | None = None,
    include_baselines: Iterable[str] = (),
    allow_missing_as_abstain: bool = False,
    max_cases: int = 20,
) -> dict[str, Any]:
    candidates = _load_jsonl_map(candidates_path)
    methods: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in selections.items():
        method_name = str(name).strip()
        if not method_name:
            raise ValueError("selection method names must be nonempty")
        if method_name in methods:
            raise ValueError(f"duplicate selection method name: {method_name}")
        methods[method_name] = _load_jsonl_map(path)

    abstentions = abstentions or {}
    for name, path in abstentions.items():
        method_name = str(name).strip()
        if method_name not in methods:
            raise ValueError(
                f"abstention file supplied for unknown selection method: {method_name!r}"
            )
        abstention_rows = _load_jsonl_map(path)
        duplicate = sorted(set(methods[method_name]) & set(abstention_rows))
        if duplicate:
            raise ValueError(
                f"method {method_name!r} has problem_ids in both selection and "
                f"abstention files: {', '.join(duplicate[:5])}"
            )
        methods[method_name] = {**methods[method_name], **abstention_rows}

    for baseline in include_baselines:
        name = str(baseline).strip()
        if not name:
            continue
        if name not in DERIVED_BASELINES:
            raise ValueError(f"unsupported derived baseline={name!r}")
        if name in methods:
            raise ValueError(f"duplicate selection method name: {name}")
        methods[name] = _derived_selection_map(candidates, name)

    if len(methods) < 2:
        raise ValueError("at least two selection methods are required")

    candidate_ids = _nonempty_candidate_ids(candidates)
    problem_ids = _problem_ids(
        candidate_ids,
        methods,
        allow_missing_as_abstain=allow_missing_as_abstain,
    )
    if not problem_ids:
        raise ValueError("no comparable problem_ids found")

    labels_by_id: dict[str, list[int]] = {}
    candidates_by_id: dict[str, list[str]] = {}
    any_correct: dict[str, bool] = {}
    total_candidates = 0
    total_candidate_pairs = 0
    for problem_id in problem_ids:
        grouped = validate_grouped_candidates(candidates[problem_id], require_labels=True)
        assert grouped.labels is not None
        labels = [1 if int(label) else 0 for label in grouped.labels]
        if not grouped.candidates:
            continue
        labels_by_id[problem_id] = labels
        candidates_by_id[problem_id] = list(grouped.candidates)
        any_correct[problem_id] = any(bool(label) for label in labels)
        total_candidates += len(labels)
        total_candidate_pairs += len(labels) * (len(labels) - 1) // 2

    problem_ids = [problem_id for problem_id in problem_ids if problem_id in labels_by_id]
    if not problem_ids:
        raise ValueError("no non-empty candidate sets found")

    decisions = {
        name: _method_decisions(
            name=name,
            records=records,
            problem_ids=problem_ids,
            labels_by_id=labels_by_id,
            any_correct=any_correct,
            allow_missing_as_abstain=allow_missing_as_abstain,
        )
        for name, records in methods.items()
    }
    method_summaries = {
        name: _method_summary(rows, any_correct=any_correct)
        for name, rows in decisions.items()
    }
    ordered_names = sorted(
        method_summaries,
        key=lambda name: (
            -method_summaries[name]["selected_correct"]["successes"],
            -method_summaries[name]["accepted_accuracy"]["rate"],
            name,
        ),
    )
    best_name = ordered_names[0]
    pairwise = {
        a: {
            b: paired_comparison(
                [row["selected_correct"] for row in decisions[a]],
                [row["selected_correct"] for row in decisions[b]],
            ).to_json_dict()
            for b in ordered_names
            if b != a
        }
        for a in ordered_names
    }

    best_cases = {
        name: _top_disagreements(
            challenger=best_name,
            baseline=name,
            challenger_rows=decisions[best_name],
            baseline_rows=decisions[name],
            candidates_by_id=candidates_by_id,
            max_cases=max_cases,
        )
        for name in ordered_names
        if name != best_name
    }

    any_total = sum(1 for value in any_correct.values() if value)
    return {
        "dataset": {
            "problems": len(problem_ids),
            "has_any_correct": proportion_summary(any_total, len(problem_ids)).to_json_dict(),
            "avg_candidates_per_problem": total_candidates / float(len(problem_ids)),
            "total_candidate_pairs": int(total_candidate_pairs),
            "allow_missing_as_abstain": bool(allow_missing_as_abstain),
        },
        "methods": {name: method_summaries[name] for name in ordered_names},
        "best_method": best_name,
        "coverage_accuracy_frontier": _coverage_accuracy_frontier(method_summaries),
        "pairwise": pairwise,
        "best_method_cases": best_cases,
        "config": {
            "selection_methods": list(selections),
            "abstention_methods": list(abstentions),
            "derived_baselines": [str(name) for name in include_baselines],
            "max_cases": int(max(0, max_cases)),
        },
    }


def format_markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    methods = summary["methods"]
    best = summary["best_method"]
    lines = [
        "# Selection Leaderboard",
        "",
        f"- Problems: {dataset['problems']}",
        f"- Any-correct ceiling: {_fmt_prop(dataset['has_any_correct'])}",
        f"- Avg candidates/problem: {dataset['avg_candidates_per_problem']:.2f}",
        f"- Best method: `{best}`",
        "",
        "## Leaderboard",
        "",
        "| method | coverage | selected correct | accepted accuracy | correct given any | missed available | abstained |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in methods.items():
        lines.append(
            f"| `{_md(name)}` | {_pct(row['coverage'])} | "
            f"{_fmt_prop(row['selected_correct'])} | "
            f"{_fmt_prop(row['accepted_accuracy'])} | "
            f"{_fmt_prop(row['selected_correct_given_any'])} | "
            f"{row['missed_available_correct']} | {row['abstained']} |"
        )

    frontier = summary.get("coverage_accuracy_frontier", [])
    lines.extend(
        [
            "",
            "## Coverage/Accuracy Frontier",
            "",
            "| method | coverage | accepted accuracy | selected correct | abstained |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in frontier:
        method = methods[row["method"]]
        lines.append(
            f"| `{_md(row['method'])}` | {_pct(row['coverage'])} | "
            f"{_fmt_prop(row['accepted_accuracy'])} | "
            f"{_fmt_prop(method['selected_correct'])} | {method['abstained']} |"
        )

    lines.extend(
        [
            "",
            f"## Paired Against `{_md(best)}`",
            "",
            "| compared method | both correct | compared only | best only | best lift | discordant | p-value |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in methods:
        if name == best:
            continue
        row = summary["pairwise"][name][best]
        lines.append(
            f"| `{_md(name)}` | {row['both_success']} | {row['a_only']} | "
            f"{row['b_only']} | {_pct(row['b_minus_a'])} | "
            f"{row['discordant']} | {row['exact_sign_p']:.6g} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Lift Matrix",
            "",
            "Rows are baselines; columns are compared methods. Values are compared minus row accuracy.",
            "",
        ]
    )
    names = list(methods)
    lines.append("| baseline \\ compared | " + " | ".join(f"`{_md(name)}`" for name in names) + " |")
    lines.append("|---" + "|---:" * len(names) + "|")
    for a in names:
        values = []
        for b in names:
            if a == b:
                values.append("-")
            else:
                values.append(_pct(summary["pairwise"][a][b]["b_minus_a"]))
        lines.append(f"| `{_md(a)}` | " + " | ".join(values) + " |")

    for name, cases in summary["best_method_cases"].items():
        _append_case_table(lines, best, name, cases)
    return "\n".join(lines).rstrip()


def _method_decisions(
    *,
    name: str,
    records: dict[str, dict[str, Any]],
    problem_ids: list[str],
    labels_by_id: dict[str, list[int]],
    any_correct: dict[str, bool],
    allow_missing_as_abstain: bool,
) -> list[dict[str, Any]]:
    rows = []
    for problem_id in problem_ids:
        labels = labels_by_id[problem_id]
        record = records.get(problem_id)
        if record is None:
            if not allow_missing_as_abstain:
                raise ValueError(f"method {name!r} is missing problem_id={problem_id!r}")
            rows.append(_decision(problem_id, accepted=False, selected_correct=False))
            continue
        if _is_abstention(record):
            rows.append(
                _decision(
                    problem_id,
                    accepted=False,
                    selected_correct=False,
                    chosen_index=_optional_index(record),
                )
            )
            continue
        chosen_index = _chosen_index(record, problem_id=problem_id, n_labels=len(labels))
        selected_correct = bool(labels[chosen_index])
        rows.append(
            _decision(
                problem_id,
                accepted=True,
                selected_correct=selected_correct,
                chosen_index=chosen_index,
                has_any_correct=any_correct[problem_id],
            )
        )
    return rows


def _method_summary(
    rows: list[dict[str, Any]],
    *,
    any_correct: dict[str, bool],
) -> dict[str, Any]:
    total = len(rows)
    accepted = sum(1 for row in rows if row["accepted"])
    abstained = total - accepted
    selected_correct = sum(1 for row in rows if row["selected_correct"])
    accepted_correct = sum(
        1 for row in rows if row["accepted"] and row["selected_correct"]
    )
    any_total = sum(1 for row in rows if any_correct[row["problem_id"]])
    selected_given_any = sum(
        1
        for row in rows
        if any_correct[row["problem_id"]] and row["selected_correct"]
    )
    missed_available = sum(
        1
        for row in rows
        if any_correct[row["problem_id"]] and not row["selected_correct"]
    )
    return {
        "problems": int(total),
        "accepted": int(accepted),
        "abstained": int(abstained),
        "coverage": accepted / float(total) if total else 0.0,
        "selected_correct": proportion_summary(selected_correct, total).to_json_dict(),
        "accepted_accuracy": proportion_summary(accepted_correct, accepted).to_json_dict(),
        "selected_correct_given_any": proportion_summary(selected_given_any, any_total).to_json_dict(),
        "missed_available_correct": int(missed_available),
    }


def _coverage_accuracy_frontier(
    method_summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    frontier = []
    for name, row in method_summaries.items():
        if int(row["accepted"]) == 0:
            continue
        coverage = float(row["coverage"])
        accuracy = float(row["accepted_accuracy"]["rate"])
        dominated = False
        for other_name, other in method_summaries.items():
            if other_name == name or int(other["accepted"]) == 0:
                continue
            other_coverage = float(other["coverage"])
            other_accuracy = float(other["accepted_accuracy"]["rate"])
            if (
                other_coverage >= coverage
                and other_accuracy >= accuracy
                and (other_coverage > coverage or other_accuracy > accuracy)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(
                {
                    "method": name,
                    "coverage": coverage,
                    "accepted_accuracy": row["accepted_accuracy"],
                    "accepted": int(row["accepted"]),
                }
            )
    return sorted(
        frontier,
        key=lambda row: (-float(row["coverage"]), -float(row["accepted_accuracy"]["rate"]), row["method"]),
    )


def _top_disagreements(
    *,
    challenger: str,
    baseline: str,
    challenger_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    candidates_by_id: dict[str, list[str]],
    max_cases: int,
) -> list[dict[str, Any]]:
    cases = []
    for left, right in zip(challenger_rows, baseline_rows):
        if left["selected_correct"] and not right["selected_correct"]:
            problem_id = left["problem_id"]
            cases.append(
                {
                    "problem_id": problem_id,
                    "challenger": challenger,
                    "baseline": baseline,
                    "challenger_accepted": bool(left["accepted"]),
                    "baseline_accepted": bool(right["accepted"]),
                    "challenger_index": left.get("chosen_index"),
                    "baseline_index": right.get("chosen_index"),
                    "challenger_candidate": _candidate_text(
                        candidates_by_id[problem_id],
                        left.get("chosen_index"),
                    ),
                    "baseline_candidate": _candidate_text(
                        candidates_by_id[problem_id],
                        right.get("chosen_index"),
                    ),
                }
            )
    cases.sort(key=lambda row: row["problem_id"])
    return cases[: max(0, int(max_cases))]


def _candidate_text(candidates: list[str], index: Any) -> str:
    if index is None:
        return ""
    idx = int(index)
    if idx < 0 or idx >= len(candidates):
        return ""
    return candidates[idx]


def _decision(
    problem_id: str,
    *,
    accepted: bool,
    selected_correct: bool,
    chosen_index: int | None = None,
    has_any_correct: bool = False,
) -> dict[str, Any]:
    return {
        "problem_id": problem_id,
        "accepted": bool(accepted),
        "selected_correct": bool(selected_correct),
        "chosen_index": chosen_index,
        "has_any_correct": bool(has_any_correct),
    }


def _load_jsonl_map(path: str | Path) -> dict[str, dict[str, Any]]:
    return load_jsonl_map_by_problem_id(path, encoding="utf-8-sig")


def _derived_selection_map(
    candidates: dict[str, dict[str, Any]],
    baseline: str,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for problem_id, row in candidates.items():
        grouped = validate_grouped_candidates(row, require_labels=True)
        if not grouped.candidates:
            continue
        if baseline == "first":
            chosen = 0
        elif baseline == "shortest":
            normalized = [normalize_lean_statement(text) for text in grouped.candidates]
            chosen = min(range(len(normalized)), key=lambda idx: (len(normalized[idx]), idx))
        else:
            raise ValueError(f"unsupported derived baseline={baseline!r}")
        out[problem_id] = {
            "problem_id": problem_id,
            "chosen_index": int(chosen),
            "selection_method": baseline,
        }
    return out


def _problem_ids(
    candidate_ids: set[str],
    methods: dict[str, dict[str, dict[str, Any]]],
    *,
    allow_missing_as_abstain: bool,
) -> list[str]:
    if allow_missing_as_abstain:
        for name, records in sorted(methods.items()):
            extra = sorted(set(records) - candidate_ids)
            if extra:
                raise ValueError(
                    f"method {name!r} has problem_ids missing from candidates: "
                    + ", ".join(repr(pid) for pid in extra[:5])
                )
        return sorted(candidate_ids)
    return matching_problem_ids_many(
        {
            "candidates": {problem_id: {} for problem_id in candidate_ids},
            **{f"method:{name}": records for name, records in methods.items()},
        }
    )


def _nonempty_candidate_ids(candidates: dict[str, dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for problem_id, row in candidates.items():
        grouped = validate_grouped_candidates(row, require_labels=True)
        if grouped.candidates:
            ids.add(problem_id)
    return ids


def _is_abstention(record: dict[str, Any]) -> bool:
    return record.get("abstained") is True or record.get("accepted") is False


def _optional_index(record: dict[str, Any]) -> int | None:
    if "chosen_index" in record:
        return int(record["chosen_index"])
    raw = record.get("chosen_indices")
    if isinstance(raw, list) and raw:
        return int(raw[0])
    return None


def _chosen_index(record: dict[str, Any], *, problem_id: str, n_labels: int) -> int:
    chosen = _optional_index(record)
    if chosen is None:
        raise ValueError(f"selection record has no chosen_index for {problem_id}: {record!r}")
    if chosen < 0 or chosen >= n_labels:
        raise ValueError(
            f"chosen index out of range for {problem_id}: {chosen} (n={n_labels})"
        )
    return int(chosen)


def _append_case_table(
    lines: list[str],
    best: str,
    baseline: str,
    cases: list[dict[str, Any]],
) -> None:
    lines.extend(["", f"## `{_md(best)}` Wins Over `{_md(baseline)}`", ""])
    if not cases:
        lines.append("No cases.")
        return
    lines.append(
        "| problem_id | best status | baseline status | best index | baseline index | best candidate | baseline candidate |"
    )
    lines.append("|---|---|---|---:|---:|---|---|")
    for row in cases:
        lines.append(
            f"| {_md(row['problem_id'])} | {_status(row['challenger_accepted'])} | "
            f"{_status(row['baseline_accepted'])} | "
            f"{_fmt_index(row['challenger_index'])} | "
            f"{_fmt_index(row['baseline_index'])} | "
            f"{_md(_short(row['challenger_candidate']))} | "
            f"{_md(_short(row['baseline_candidate']))} |"
        )


def _fmt_prop(row: dict[str, Any]) -> str:
    return f"{_pct(row['rate'])} ({row['successes']}/{row['total']})"


def _fmt_index(value: Any) -> str:
    return "-" if value is None else str(int(value))


def _status(accepted: bool) -> str:
    return "accepted" if accepted else "abstained"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _short(value: object, max_len: int = 90) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _md(value: object) -> str:
    return str(value).replace("|", "\\|")


def _parse_selection(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("selections must be NAME=PATH")
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("selections must be NAME=PATH")
    return name, path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument(
        "--selection",
        action="append",
        default=[],
        type=_parse_selection,
        help="Selection file as NAME=PATH. Repeat for multiple methods.",
    )
    parser.add_argument(
        "--abstention",
        action="append",
        default=[],
        type=_parse_selection,
        help="Optional abstention file as NAME=PATH for an existing --selection method.",
    )
    parser.add_argument(
        "--include-baseline",
        action="append",
        default=[],
        choices=tuple(sorted(DERIVED_BASELINES)),
        help="Add a derived baseline from candidates.",
    )
    parser.add_argument("--allow-missing-as-abstain", action="store_true")
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    summary = analyze_leaderboard(
        candidates_path=args.candidates,
        selections=dict(args.selection),
        abstentions=dict(args.abstention),
        include_baselines=list(args.include_baseline),
        allow_missing_as_abstain=bool(args.allow_missing_as_abstain),
        max_cases=int(args.max_cases),
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
