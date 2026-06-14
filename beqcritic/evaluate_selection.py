"""
CLI: evaluate selection outputs against grouped candidate labels.

Inputs:
  - candidates JSONL from `beqcritic.make_grouped_candidates` (must include `labels`)
  - selections JSONL from `beqcritic.score_and_select`

Abstaining selectors are supported through explicit rows with
`abstained: true` / `accepted: false`, an optional abstentions JSONL, or
`--treat-missing-as-abstain` for accepted-only output files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .statistics import proportion_summary
from .textnorm import normalize_lean_statement


def _load_jsonl(path: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = obj.get("problem_id")
            if pid is None:
                raise ValueError(f"Missing problem_id in {path}: {obj}")
            out[str(pid)] = obj
    return out


def _merge_selection_maps(
    selections: dict[str, dict[str, Any]],
    abstentions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    duplicate = sorted(set(selections) & set(abstentions))
    if duplicate:
        raise ValueError(
            "problem_ids present in both selections and abstentions: "
            + ", ".join(duplicate[:5])
        )
    merged = dict(selections)
    merged.update(abstentions)
    return merged


def _is_abstention_row(row: dict[str, Any]) -> bool:
    return row.get("abstained") is True or row.get("accepted") is False


def _chosen_indices(
    row: dict[str, Any],
    *,
    problem_id: str,
    n_labels: int,
    required: bool,
) -> list[int]:
    chosen_indices_raw = row.get("chosen_indices", None)
    if isinstance(chosen_indices_raw, list) and chosen_indices_raw:
        chosen_indices = [int(x) for x in chosen_indices_raw]
    elif "chosen_index" in row:
        chosen_indices = [int(row.get("chosen_index"))]
    elif required:
        raise ValueError(f"Missing chosen_index in selections for {problem_id}: {row}")
    else:
        return []

    bad = [i for i in chosen_indices if i < 0 or i >= n_labels]
    if bad:
        bad.sort()
        raise ValueError(
            f"chosen_indices out of range for {problem_id}: {bad[:5]} (n={n_labels})"
        )
    return chosen_indices


def _prop(successes: int, total: int) -> dict[str, float | int]:
    return proportion_summary(int(successes), int(total)).to_json_dict()


def evaluate_selection_records(
    candidates_by_id: dict[str, dict[str, Any]],
    selections_by_id: dict[str, dict[str, Any]],
    *,
    max_k: int = 1,
    treat_missing_as_abstain: bool = False,
) -> dict[str, Any]:
    """Return selection metrics as a JSON-serializable summary."""
    max_k = max(1, int(max_k))
    missing_cand = sorted(set(selections_by_id) - set(candidates_by_id))
    missing_sel = sorted(set(candidates_by_id) - set(selections_by_id))
    has_explicit_abstentions = any(
        _is_abstention_row(row)
        for pid, row in selections_by_id.items()
        if pid in candidates_by_id
    )
    abstention_mode = bool(treat_missing_as_abstain or has_explicit_abstentions)
    if treat_missing_as_abstain:
        pids = sorted(candidates_by_id)
    else:
        pids = sorted(set(candidates_by_id) & set(selections_by_id))

    problems = 0
    n_any = 0
    accepted = 0
    accepted_any = 0
    accepted_correct = 0
    accepted_correct_any = 0
    abstained = 0
    explicit_abstained = 0
    missing_as_abstained = 0
    abstained_any = 0
    abstained_with_choice = 0
    abstained_selected_correct = 0
    topk_correct = [0 for _ in range(max_k)]
    topk_correct_any = [0 for _ in range(max_k)]
    n_first_correct = 0
    n_shortest_correct = 0
    comp_sizes: list[int] = []
    comp_cohesions: list[float] = []
    chosen_centralities: list[float] = []
    chosen_centrality_gaps: list[float] = []
    edges_before: list[int] = []
    edges_after: list[int] = []
    components_before: list[int] = []
    components_after: list[int] = []
    isolated_before: list[int] = []
    isolated_after: list[int] = []
    edges_readded: list[int] = []

    for pid in pids:
        c = candidates_by_id[pid]
        s = selections_by_id.get(pid)

        candidate_texts = c.get("candidates") or []
        labels = c.get("labels") or []
        if len(candidate_texts) != len(labels):
            raise ValueError(
                f"Candidates/labels length mismatch for {pid}: "
                f"{len(candidate_texts)} vs {len(labels)}"
            )
        if not candidate_texts:
            continue

        labels01 = [1 if int(x) else 0 for x in labels]
        any_correct = any(labels01)
        problems += 1
        n_any += int(any_correct)
        n_first_correct += int(bool(labels01[0]))

        norm = [normalize_lean_statement(x) for x in candidate_texts]
        shortest_idx = min(range(len(norm)), key=lambda i: (len(norm[i]), i))
        n_shortest_correct += int(bool(labels01[shortest_idx]))

        if s is None:
            if not treat_missing_as_abstain:
                continue
            abstained += 1
            missing_as_abstained += 1
            abstained_any += int(any_correct)
            continue

        if s.get("accepted") is True and s.get("abstained") is True:
            raise ValueError(f"Conflicting accepted/abstained flags for {pid}: {s}")

        is_abstention = _is_abstention_row(s)
        chosen_indices = _chosen_indices(
            s,
            problem_id=pid,
            n_labels=len(labels01),
            required=not is_abstention,
        )

        if is_abstention:
            abstained += 1
            explicit_abstained += 1
            abstained_any += int(any_correct)
            if chosen_indices:
                abstained_with_choice += 1
                abstained_selected_correct += int(bool(labels01[int(chosen_indices[0])]))
            continue

        accepted += 1
        accepted_any += int(any_correct)
        chosen_index = int(chosen_indices[0])
        sel_correct = bool(labels01[chosen_index])
        accepted_correct += int(sel_correct)
        accepted_correct_any += int(sel_correct and any_correct)

        for k in range(1, max_k + 1):
            picked = chosen_indices[:k]
            hit = any(bool(labels01[i]) for i in picked)
            topk_correct[k - 1] += int(hit)
            topk_correct_any[k - 1] += int(hit and any_correct)

        if "component_size" in s:
            comp_sizes.append(int(s["component_size"]))
        if "component_cohesion" in s:
            comp_cohesions.append(float(s["component_cohesion"]))
        if "chosen_centrality" in s:
            chosen_centralities.append(float(s["chosen_centrality"]))
        if "chosen_centrality_gap" in s:
            chosen_centrality_gaps.append(float(s["chosen_centrality_gap"]))
        if "edges_before" in s:
            edges_before.append(int(s["edges_before"]))
        if "edges_after" in s:
            edges_after.append(int(s["edges_after"]))
        if "components_before" in s:
            components_before.append(int(s["components_before"]))
        if "components_after" in s:
            components_after.append(int(s["components_after"]))
        if "isolated_before" in s:
            isolated_before.append(int(s["isolated_before"]))
        if "isolated_after" in s:
            isolated_after.append(int(s["isolated_after"]))
        if "edges_readded" in s:
            edges_readded.append(int(s["edges_readded"]))

    if problems == 0:
        raise ValueError("No problem_ids to evaluate.")

    averages: dict[str, float] = {}
    if comp_sizes:
        averages["component_size"] = sum(comp_sizes) / len(comp_sizes)
    if comp_cohesions:
        averages["component_cohesion"] = sum(comp_cohesions) / len(comp_cohesions)
    if chosen_centralities:
        averages["chosen_centrality"] = sum(chosen_centralities) / len(chosen_centralities)
    if chosen_centrality_gaps:
        averages["chosen_centrality_gap"] = sum(chosen_centrality_gaps) / len(chosen_centrality_gaps)
    if edges_before:
        averages["edges_before"] = sum(edges_before) / len(edges_before)
    if edges_after:
        averages["edges_after"] = sum(edges_after) / len(edges_after)
    if components_before:
        averages["components_before"] = sum(components_before) / len(components_before)
    if components_after:
        averages["components_after"] = sum(components_after) / len(components_after)
    if isolated_before:
        averages["isolated_before"] = sum(isolated_before) / len(isolated_before)
    if isolated_after:
        averages["isolated_after"] = sum(isolated_after) / len(isolated_after)
    if edges_readded:
        averages["edges_readded"] = sum(edges_readded) / len(edges_readded)

    accepted_selected = _prop(accepted_correct, accepted)
    summary: dict[str, Any] = {
        "problems": int(problems),
        "selection_records": int(len(selections_by_id)),
        "abstention_mode": bool(abstention_mode),
        "accepted": int(accepted),
        "abstained": int(abstained),
        "explicit_abstained": int(explicit_abstained),
        "missing_as_abstained": int(missing_as_abstained),
        "coverage": _prop(accepted, problems),
        "has_any_correct": _prop(n_any, problems),
        "accepted_has_any_correct": _prop(accepted_any, accepted),
        "accepted_selected_correct": accepted_selected,
        "accepted_selected_correct_given_any": _prop(accepted_correct_any, accepted_any),
        "selected_correct": accepted_selected,
        "selected_correct_given_any": _prop(accepted_correct_any, accepted_any),
        "selected_correct_counting_abstentions_incorrect": _prop(accepted_correct, problems),
        "selected_correct_with_abstention_choices": _prop(
            accepted_correct + abstained_selected_correct,
            accepted + abstained_with_choice,
        ),
        "abstained_has_any_correct": _prop(abstained_any, abstained),
        "abstained_selected_correct": _prop(abstained_selected_correct, abstained_with_choice),
        "abstained_with_choice": int(abstained_with_choice),
        "topk_any_correct": [_prop(value, accepted) for value in topk_correct],
        "topk_any_correct_given_any": [
            _prop(value, accepted_any) for value in topk_correct_any
        ],
        "baselines": {
            "first": _prop(n_first_correct, problems),
            "shortest": _prop(n_shortest_correct, problems),
        },
        "averages": averages,
        "warnings": {
            "selections_missing_candidates": int(len(missing_cand)),
            "candidates_missing_selections": int(len(missing_sel)),
            "selection_ids_without_candidates": missing_cand[:20],
            "candidate_ids_without_selections": missing_sel[:20],
        },
    }
    return summary


def _pct(successes: int, total: int) -> float:
    return 100.0 * int(successes) / max(1, int(total))


def _format_count(metric: dict[str, Any]) -> str:
    return f"{metric['successes']} ({_pct(int(metric['successes']), int(metric['total'])):.1f}%)"


def format_summary(summary: dict[str, Any]) -> str:
    problems = int(summary["problems"])
    accepted = int(summary["accepted"])
    abstained = int(summary["abstained"])
    has_any = summary["has_any_correct"]
    selected = summary["accepted_selected_correct"]
    selected_any = summary["accepted_selected_correct_given_any"]
    baselines = summary["baselines"]
    topk = summary["topk_any_correct"]
    topk_any = summary["topk_any_correct_given_any"]
    averages = summary["averages"]

    lines: list[str] = [f"Problems: {problems}"]
    if summary["abstention_mode"]:
        lines.extend(
            [
                f"Accepted: {accepted} ({_pct(accepted, problems):.1f}%)",
                f"Abstained: {abstained} ({_pct(abstained, problems):.1f}%)",
            ]
        )
        if int(summary["explicit_abstained"]):
            lines.append(f"Explicit abstentions: {summary['explicit_abstained']}")
        if int(summary["missing_as_abstained"]):
            lines.append(
                "Missing selections counted as abstentions: "
                f"{summary['missing_as_abstained']}"
            )
        lines.append(f"Has any correct: {_format_count(has_any)}")
        lines.append(f"Accepted selected correct: {_format_count(selected)}")
        if int(summary["accepted_has_any_correct"]["successes"]):
            lines.append(
                "Accepted selected correct | accepted any correct: "
                f"{_format_count(selected_any)}"
            )
        lines.append(
            "Selected correct with abstentions counted as incorrect: "
            f"{_format_count(summary['selected_correct_counting_abstentions_incorrect'])}"
        )
        if abstained:
            lines.append(
                "Abstained with any correct: "
                f"{_format_count(summary['abstained_has_any_correct'])}"
            )
        if int(summary["abstained_with_choice"]):
            lines.append(
                "Abstained selected correct (diagnostic): "
                f"{_format_count(summary['abstained_selected_correct'])}"
            )
            lines.append(
                "Full-coverage selected correct where abstention choices are known: "
                f"{_format_count(summary['selected_correct_with_abstention_choices'])}"
            )
    else:
        lines.append(f"Has any correct: {_format_count(has_any)}")
        lines.append(f"Selected correct: {_format_count(selected)}")
        if int(has_any["successes"]):
            lines.append(f"Selected correct | any correct: {_format_count(selected_any)}")

    if len(topk) > 1:
        label_prefix = "Accepted top" if summary["abstention_mode"] else "Top"
        for k in range(2, len(topk) + 1):
            lines.append(f"{label_prefix}-{k} any correct: {_format_count(topk[k - 1])}")
            if int(summary["accepted_has_any_correct"]["successes"]):
                lines.append(
                    f"{label_prefix}-{k} any correct | any correct: "
                    f"{_format_count(topk_any[k - 1])}"
                )
    lines.append(f"Baseline first: {_format_count(baselines['first'])}")
    lines.append(f"Baseline shortest: {_format_count(baselines['shortest'])}")
    if "component_size" in averages:
        lines.append(f"Avg component_size: {averages['component_size']:.2f}")
    if "component_cohesion" in averages:
        lines.append(f"Avg component_cohesion: {averages['component_cohesion']:.3f}")
    if "chosen_centrality" in averages:
        lines.append(f"Avg chosen_centrality: {averages['chosen_centrality']:.3f}")
    if "chosen_centrality_gap" in averages:
        lines.append(f"Avg chosen_centrality_gap: {averages['chosen_centrality_gap']:.3f}")
    if "edges_before" in averages and "edges_after" in averages:
        lines.append(
            "Avg edges_before/after: "
            f"{averages['edges_before']:.1f} -> {averages['edges_after']:.1f}"
        )
    if "components_before" in averages and "components_after" in averages:
        lines.append(
            "Avg components_before/after: "
            f"{averages['components_before']:.2f} -> {averages['components_after']:.2f}"
        )
    if "isolated_before" in averages and "isolated_after" in averages:
        lines.append(
            "Avg isolated_before/after: "
            f"{averages['isolated_before']:.2f} -> {averages['isolated_after']:.2f}"
        )
    if "edges_readded" in averages:
        lines.append(f"Avg edges_readded: {averages['edges_readded']:.2f}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=str, required=True)
    p.add_argument("--selections", type=str, required=True)
    p.add_argument(
        "--abstentions",
        type=str,
        default="",
        help="Optional JSONL of explicit abstention rows to merge with selections.",
    )
    p.add_argument(
        "--treat-missing-as-abstain",
        action="store_true",
        help="Count candidate problem_ids absent from selections as abstentions.",
    )
    p.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional path for a machine-readable metric summary.",
    )
    p.add_argument(
        "--max-k",
        type=int,
        default=1,
        help="If selections contain chosen_indices, also report top-k any-correct metrics up to this k.",
    )
    args = p.parse_args()

    cand = _load_jsonl(args.candidates)
    sel = _load_jsonl(args.selections)
    if args.abstentions:
        sel = _merge_selection_maps(sel, _load_jsonl(args.abstentions))

    try:
        summary = evaluate_selection_records(
            cand,
            sel,
            max_k=int(args.max_k),
            treat_missing_as_abstain=bool(args.treat_missing_as_abstain),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    warnings = summary["warnings"]
    if int(warnings["selections_missing_candidates"]):
        print(f"Warning: {warnings['selections_missing_candidates']} selections missing candidates")
    if int(warnings["candidates_missing_selections"]):
        suffix = " (counted as abstentions)" if args.treat_missing_as_abstain else ""
        print(f"Warning: {warnings['candidates_missing_selections']} candidates missing selections{suffix}")

    print(format_summary(summary))
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
