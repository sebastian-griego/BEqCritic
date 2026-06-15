#!/usr/bin/env python3
"""Generate paper-ready NLVerifier metrics from checked-in result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SPLITS = {
    "transductive": (
        ("NL (full)", "transductive_nl"),
        ("NL blank", "transductive_nl_blank"),
        ("NL const", "transductive_nl_const"),
    ),
    "inductive": (
        ("NL (full)", "inductive_nl"),
        ("NL blank", "inductive_nl_blank"),
        ("NL const", "inductive_nl_const"),
    ),
}

BASELINE_METRICS = {
    "transductive": {
        "random": "exp_transductive/metrics_random.json",
        "self_bleu": "exp_transductive/metrics_self_bleu.json",
        "critic_global_medoid": "exp_transductive/metrics_critic_global_medoid.json",
    },
    "inductive": {
        "random": "exp_inductive/metrics_random.json",
        "self_bleu": "exp_inductive/metrics_self_bleu.json",
        "critic_global_medoid": "exp_inductive/metrics_critic_global_medoid.json",
    },
}

MAIN_TABLE_ROWS = (
    {
        "key": "random",
        "label": r"Random selection",
        "source": "baseline",
    },
    {
        "key": "self_bleu",
        "label": r"Self-BLEU consensus \citep{zhu2018texygen,poiroux2025reliable}",
        "source": "baseline",
    },
    {
        "key": "critic_global_medoid",
        "label": r"Candidate-only critic (best variant)",
        "source": "baseline",
    },
    {
        "key": "nl_full",
        "label": r"NLVerifier (full input)",
        "source": "proofnetverif",
        "proofnetverif_keys": {
            "transductive": "transductive_nl",
            "inductive": "inductive_nl",
        },
        "bold": True,
        "midrule_before": True,
    },
    {
        "key": "nl_blank",
        "label": r"NLVerifier (blank NL)",
        "source": "proofnetverif",
        "proofnetverif_keys": {
            "transductive": "transductive_nl_blank",
            "inductive": "inductive_nl_blank",
        },
    },
    {
        "key": "nl_const",
        "label": r"NLVerifier (constant NL)",
        "source": "proofnetverif",
        "proofnetverif_keys": {
            "transductive": "transductive_nl_const",
            "inductive": "inductive_nl_const",
        },
    },
)


def build_summary(
    results_dir: Path,
    *,
    ablation_json: Path | None = None,
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    ablation_path = ablation_json or results_dir / "nlverifier_proofnetverif_ablation_metrics.json"
    confidence_path = results_dir / "exp_inductive" / "nlverifier_confidence_audit.json"
    abstention_path = results_dir / "exp_inductive" / "metrics_nlverifier_abstain_p50.json"
    stability_path = results_dir / "exp_inductive" / "nlverifier_threshold_stability_p50.json"
    leaderboard_path = results_dir / "exp_inductive" / "selection_leaderboard.json"
    ood_path = results_dir / "ood_formalalign_minif2f.json"
    baseline_paths = {
        split: {
            name: results_dir / rel_path
            for name, rel_path in split_paths.items()
        }
        for split, split_paths in BASELINE_METRICS.items()
    }
    source_paths = {
        "proofnetverif_ablation": ablation_path,
        "confidence_audit": confidence_path,
        "abstention_metrics": abstention_path,
        "threshold_stability": stability_path,
        "selection_leaderboard": leaderboard_path,
        "ood_formalalign": ood_path,
        **{
            f"{split}_{name}": path
            for split, split_paths in baseline_paths.items()
            for name, path in split_paths.items()
        },
    }

    ablation = _load_json(ablation_path)
    confidence = _load_json(confidence_path)
    abstention = _load_json(abstention_path)
    stability = _load_json(stability_path)
    leaderboard = _load_json(leaderboard_path)
    ood = _load_json(ood_path)
    baselines = {
        split: {
            name: _load_json(path)
            for name, path in split_paths.items()
        }
        for split, split_paths in baseline_paths.items()
    }

    settings = dict(ablation["settings"])
    _validate_source_consistency(
        settings=settings,
        baselines=baselines,
        confidence=confidence,
        abstention=abstention,
        stability=stability,
        leaderboard=leaderboard,
        ood=ood,
    )
    proofnetverif = {
        split: {
            key: settings[key]
            for _, key in rows
        }
        for split, rows in SPLITS.items()
    }
    confidence_signals = _confidence_signal_summary(confidence)
    abstention_policy = _abstention_summary(abstention, stability, leaderboard)
    leaderboard_summary = _leaderboard_summary(leaderboard)
    main_table = _main_table_summary(proofnetverif, baselines)

    return {
        **settings,
        "provenance": {
            **dict(ablation.get("provenance", {})),
            "generator": "scripts/summarize_nlverifier_paper_metrics.py",
            "sources": {
                name: _rel(path)
                for name, path in source_paths.items()
            },
            "source_sha256": {
                name: _file_sha256(path)
                for name, path in source_paths.items()
            },
        },
        "proofnetverif": proofnetverif,
        "main_table": main_table,
        "selective_prediction": {
            "confidence_signals": confidence_signals,
            "abstention_policy_p50": abstention_policy,
        },
        "leaderboard": leaderboard_summary,
        "ood_formalalign_minif2f": ood,
    }


def format_markdown(summary: dict[str, Any]) -> str:
    provenance = summary["provenance"]
    lines = [
        "# NLVerifier paper-ready metrics",
        "",
        f"All numbers are computed with NLVerifier `{provenance['model']}`.",
        f"NL-blank uses {provenance['nl_blank']}; NL-const uses: "
        f"\"{provenance['nl_const']}\"",
    ]

    _append_proofnetverif_table(
        lines,
        title="ProofNetVerif (transductive test)",
        split="transductive",
        summary=summary,
    )
    _append_proofnetverif_table(
        lines,
        title="ProofNetVerif (inductive ID-disjoint test)",
        split="inductive",
        summary=summary,
    )
    _append_selective_prediction(lines, summary["selective_prediction"])
    _append_leaderboard(lines, summary["leaderboard"])
    _append_ood(lines, summary["ood_formalalign_minif2f"])
    lines.extend(
        [
            "",
            "## Qualitative analysis",
            "See `results/exp_inductive/nlverifier_failure_cases.md` and "
            "`results/exp_inductive/nlverifier_abstention_cases_p50.md`.",
            "",
            "## Sources",
            "",
        ]
    )
    source_hashes = provenance.get("source_sha256", {})
    for name, path in provenance["sources"].items():
        digest = source_hashes.get(name)
        if digest:
            lines.append(f"- `{name}`: `{path}` (`sha256:{digest}`)")
        else:
            lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines).rstrip() + "\n"


def format_latex_main_table(summary: dict[str, Any]) -> str:
    table = summary["main_table"]
    trans = table["splits"]["transductive"]
    ind = table["splits"]["inductive"]
    lines = [
        r"% Generated by scripts/summarize_nlverifier_paper_metrics.py; do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Selection accuracy on ProofNetVerif, reported as correct/total (percent).",
        r"``Given-any'' conditions on problems whose candidate set contains at least one correct candidate",
        f"({trans['has_any_correct']}/{trans['problems']} transductive, "
        f"{ind['has_any_correct']}/{ind['problems']} inductive).}}",
        r"\label{tab:main}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        rf"& \multicolumn{{2}}{{c}}{{Transductive ({trans['problems']} problems)}} "
        rf"& \multicolumn{{2}}{{c}}{{Inductive ID-disjoint ({ind['problems']} problems)}} \\",
        r"Method & Top-1 & Given-any & Top-1 & Given-any \\",
        r"\midrule",
    ]
    for row in table["rows"]:
        if row.get("midrule_before"):
            lines.append(r"\midrule")
        cells = [
            _latex_prop(row["transductive"]["selected_correct"], bold=row.get("bold", False)),
            _latex_prop(
                row["transductive"]["selected_correct_given_any"],
                bold=row.get("bold", False),
            ),
            _latex_prop(row["inductive"]["selected_correct"], bold=row.get("bold", False)),
            _latex_prop(
                row["inductive"]["selected_correct_given_any"],
                bold=row.get("bold", False),
            ),
        ]
        lines.append(f"{row['label']} &")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _confidence_signal_summary(confidence: dict[str, Any]) -> dict[str, Any]:
    signals = {}
    for key, signal in confidence["signals"].items():
        ranking = signal["ranking_metrics"]
        best = signal["best_accuracy_prefix"]
        signals[key] = {
            "mean_prefix_accuracy": signal["mean_prefix_accuracy"],
            "mean_prefix_risk": signal["mean_prefix_risk"],
            "area_under_accuracy_coverage": ranking["area_under_accuracy_coverage"],
            "accuracy_lift_over_full": ranking["accuracy_lift_over_full"],
            "average_precision": ranking["average_precision"],
            "oracle_normalized_accuracy_area": ranking["oracle_normalized_accuracy_area"],
            "best_prefix_accuracy": best["accuracy"],
            "best_prefix_coverage": best["coverage"],
        }
    return {
        "dataset": confidence["dataset"],
        "best_by_mean_prefix_risk": confidence["best_by_mean_prefix_risk"],
        "best_by_oracle_normalized_accuracy_area": confidence[
            "best_by_oracle_normalized_accuracy_area"
        ],
        "signals": signals,
    }


def _abstention_summary(
    abstention: dict[str, Any],
    stability: dict[str, Any],
    leaderboard: dict[str, Any],
) -> dict[str, Any]:
    return {
        "confidence_key": stability["policy"]["confidence_key"],
        "target_accuracy": stability["policy"]["target_accuracy"],
        "threshold": stability["full_recommendation"]["threshold"],
        "coverage": abstention["coverage"],
        "accepted": int(abstention["accepted"]),
        "abstained": int(abstention["abstained"]),
        "accepted_selected_correct": abstention["accepted_selected_correct"],
        "accepted_selected_correct_given_any": abstention[
            "accepted_selected_correct_given_any"
        ],
        "selected_correct_counting_abstentions_incorrect": abstention[
            "selected_correct_counting_abstentions_incorrect"
        ],
        "selected_correct_with_abstention_choices": abstention[
            "selected_correct_with_abstention_choices"
        ],
        "accepted_has_any_correct": abstention["accepted_has_any_correct"],
        "threshold_stability": {
            "unique_threshold_count": stability["leave_one_out"]["unique_threshold_count"],
            "threshold_min": stability["leave_one_out"]["threshold_min"],
            "threshold_max": stability["leave_one_out"]["threshold_max"],
            "threshold_changed": stability["leave_one_out"]["threshold_changed"],
            "resamples": stability["leave_one_out"]["resamples"],
            "applied_full_accepted_min": stability["leave_one_out"][
                "applied_full_accepted_min"
            ],
            "applied_full_accepted_max": stability["leave_one_out"][
                "applied_full_accepted_max"
            ],
            "accepted_set_jaccard_min": stability["leave_one_out"][
                "accepted_set_jaccard_min"
            ],
        },
        "coverage_accuracy_frontier": leaderboard["coverage_accuracy_frontier"],
    }


def _leaderboard_summary(leaderboard: dict[str, Any]) -> dict[str, Any]:
    method_items = sorted(
        leaderboard["methods"].items(),
        key=lambda item: (
            -int(item[1]["selected_correct"]["successes"]),
            -float(item[1]["accepted_accuracy"]["rate"]),
            item[0],
        ),
    )
    methods = {
        name: {
            "coverage": row["coverage"],
            "selected_correct": row["selected_correct"],
            "accepted_accuracy": row["accepted_accuracy"],
            "missed_available_correct": row["missed_available_correct"],
            "abstained": row["abstained"],
        }
        for name, row in method_items
    }
    return {
        "dataset": leaderboard["dataset"],
        "best_method": leaderboard["best_method"],
        "methods": methods,
        "paired_against_best": leaderboard["pairwise"][leaderboard["best_method"]],
    }


def _main_table_summary(
    proofnetverif: dict[str, dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    split_meta = {
        split: _split_meta(proofnetverif[split])
        for split in ("transductive", "inductive")
    }
    rows = []
    for spec in MAIN_TABLE_ROWS:
        source = spec["source"]
        if source == "baseline":
            trans_row = baselines["transductive"][spec["key"]]
            ind_row = baselines["inductive"][spec["key"]]
        elif source == "proofnetverif":
            keys = spec["proofnetverif_keys"]
            trans_row = proofnetverif["transductive"][keys["transductive"]]
            ind_row = proofnetverif["inductive"][keys["inductive"]]
        else:
            raise ValueError(f"Unknown main-table source: {source}")
        rows.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "bold": bool(spec.get("bold", False)),
                "midrule_before": bool(spec.get("midrule_before", False)),
                "transductive": _main_table_metric(trans_row),
                "inductive": _main_table_metric(ind_row),
            }
        )
    return {"splits": split_meta, "rows": rows}


def _validate_source_consistency(
    *,
    settings: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
    confidence: dict[str, Any],
    abstention: dict[str, Any],
    stability: dict[str, Any],
    leaderboard: dict[str, Any],
    ood: dict[str, Any],
) -> None:
    for split, rows in SPLITS.items():
        first_key = rows[0][1]
        split_problems = int(settings[first_key]["problems"])
        split_has_any = _has_any_count(settings[first_key])
        for _, key in rows:
            row = settings[key]
            _validate_selection_metric_row(key, row)
            _require_equal(f"{key}.problems", int(row["problems"]), split_problems)
            _require_equal(f"{key}.has_any_correct", _has_any_count(row), split_has_any)
        for name, row in baselines[split].items():
            _validate_selection_metric_row(f"{split}_{name}", row)
            _require_equal(
                f"{split}_{name}.problems",
                int(row["problems"]),
                split_problems,
            )
            _require_equal(
                f"{split}_{name}.has_any_correct",
                _has_any_count(row),
                split_has_any,
            )

    inductive_problems = int(settings["inductive_nl"]["problems"])
    _require_equal(
        "confidence.dataset.problems",
        int(confidence["dataset"]["problems"]),
        inductive_problems,
    )
    _require_equal(
        "leaderboard.dataset.problems",
        int(leaderboard["dataset"]["problems"]),
        inductive_problems,
    )
    _require_equal(
        "abstention.accepted_plus_abstained",
        int(abstention["accepted"]) + int(abstention["abstained"]),
        inductive_problems,
    )
    _require_equal(
        "abstention.coverage.total",
        int(abstention["coverage"]["total"]),
        inductive_problems,
    )
    _require_equal(
        "abstention.coverage.successes",
        int(abstention["coverage"]["successes"]),
        int(abstention["accepted"]),
    )
    _validate_confidence_dataset(confidence["dataset"])
    _validate_confidence_signals(confidence)
    _validate_abstention_metrics(abstention)
    _validate_stability_metrics(stability, expected_problems=inductive_problems)
    _validate_leaderboard_metrics(leaderboard)
    _validate_ood_metrics(ood)


def _validate_selection_metric_row(label: str, row: dict[str, Any]) -> None:
    problems = int(row["problems"])
    has_any = _has_any_count(row)
    selected = _pct_to_count(float(row["selected_correct_pct"]), problems)
    selected_given_any = _pct_to_count(
        float(row["selected_correct_given_any_pct"]),
        has_any,
    )
    _require_bounds(f"{label}.has_any_correct", has_any, problems)
    _require_bounds(f"{label}.selected_correct", selected, problems)
    _require_bounds(f"{label}.selected_correct_given_any", selected_given_any, has_any)
    _require_equal(f"{label}.selected_correct_given_any_count", selected_given_any, selected)
    _require_pct_count(f"{label}.has_any_correct_pct", row["has_any_correct_pct"], has_any, problems)
    _require_pct_count(f"{label}.selected_correct_pct", row["selected_correct_pct"], selected, problems)
    _require_pct_count(
        f"{label}.selected_correct_given_any_pct",
        row["selected_correct_given_any_pct"],
        selected_given_any,
        has_any,
    )


def _validate_confidence_dataset(dataset: dict[str, Any]) -> None:
    problems = int(dataset["problems"])
    selected = int(dataset["selected_correct"])
    has_any = int(dataset["has_any_correct"])
    _require_bounds("confidence.dataset.selected_correct", selected, problems)
    _require_bounds("confidence.dataset.has_any_correct", has_any, problems)
    _require_rate(
        "confidence.dataset.selected_accuracy",
        dataset["selected_accuracy"],
        selected,
        problems,
    )
    _require_rate(
        "confidence.dataset.has_any_correct_rate",
        dataset["has_any_correct_rate"],
        has_any,
        problems,
    )


def _validate_confidence_signals(confidence: dict[str, Any]) -> None:
    signals = confidence["signals"]
    for label in (
        "best_by_mean_prefix_accuracy",
        "best_by_mean_prefix_risk",
        "best_by_oracle_normalized_accuracy_area",
    ):
        if label in confidence and confidence[label] not in signals:
            raise ValueError(
                f"Inconsistent source artifact confidence.{label}: "
                f"{confidence[label]!r} not in signals"
            )
    for key, signal in signals.items():
        for metric in ("mean_prefix_accuracy", "mean_prefix_risk"):
            _require_rate_bounds(f"confidence.signals.{key}.{metric}", signal[metric])
        best = signal["best_accuracy_prefix"]
        for metric in ("accuracy", "coverage"):
            _require_rate_bounds(
                f"confidence.signals.{key}.best_accuracy_prefix.{metric}",
                best[metric],
            )
        ranking = signal["ranking_metrics"]
        for metric in (
            "area_under_accuracy_coverage",
            "average_precision",
            "oracle_normalized_accuracy_area",
        ):
            _require_rate_bounds(
                f"confidence.signals.{key}.ranking_metrics.{metric}",
                ranking[metric],
            )


def _validate_abstention_metrics(abstention: dict[str, Any]) -> None:
    accepted = int(abstention["accepted"])
    abstained = int(abstention["abstained"])
    total = accepted + abstained
    _validate_prop("abstention.coverage", abstention["coverage"], successes=accepted, total=total)
    accepted_selected = int(abstention["accepted_selected_correct"]["successes"])
    accepted_has_any = int(abstention["accepted_has_any_correct"]["successes"])
    _validate_prop(
        "abstention.accepted_selected_correct",
        abstention["accepted_selected_correct"],
        total=accepted,
    )
    _validate_prop(
        "abstention.accepted_selected_correct_given_any",
        abstention["accepted_selected_correct_given_any"],
        successes=accepted_selected,
        total=accepted_has_any,
    )
    _validate_prop(
        "abstention.selected_correct_counting_abstentions_incorrect",
        abstention["selected_correct_counting_abstentions_incorrect"],
        successes=accepted_selected,
        total=total,
    )
    abstained_selected = 0
    if "abstained_selected_correct" in abstention:
        abstained_selected = int(abstention["abstained_selected_correct"]["successes"])
        _validate_prop(
            "abstention.abstained_selected_correct",
            abstention["abstained_selected_correct"],
            total=abstained,
        )
    _validate_prop(
        "abstention.selected_correct_with_abstention_choices",
        abstention["selected_correct_with_abstention_choices"],
        successes=accepted_selected + abstained_selected,
        total=total,
    )
    _validate_prop(
        "abstention.accepted_has_any_correct",
        abstention["accepted_has_any_correct"],
        total=accepted,
    )
    abstained_has_any = 0
    if "abstained_has_any_correct" in abstention:
        abstained_has_any = int(abstention["abstained_has_any_correct"]["successes"])
        _validate_prop(
            "abstention.abstained_has_any_correct",
            abstention["abstained_has_any_correct"],
            total=abstained,
        )
    if "selected_correct" in abstention:
        _validate_prop(
            "abstention.selected_correct",
            abstention["selected_correct"],
            successes=accepted_selected,
            total=accepted,
        )
    if "selected_correct_given_any" in abstention:
        _validate_prop(
            "abstention.selected_correct_given_any",
            abstention["selected_correct_given_any"],
            successes=accepted_selected,
            total=accepted_has_any,
        )
    if "has_any_correct" in abstention:
        _validate_prop(
            "abstention.has_any_correct",
            abstention["has_any_correct"],
            successes=accepted_has_any + abstained_has_any,
            total=total,
        )
    if "explicit_abstained" in abstention and "missing_as_abstained" in abstention:
        _require_equal(
            "abstention.explicit_plus_missing_abstained",
            int(abstention["explicit_abstained"]) + int(abstention["missing_as_abstained"]),
            abstained,
        )
    if "abstained_with_choice" in abstention:
        _require_equal(
            "abstention.abstained_with_choice",
            int(abstention["abstained_with_choice"]),
            abstained,
        )


def _validate_stability_metrics(stability: dict[str, Any], *, expected_problems: int) -> None:
    recommendation = stability["full_recommendation"]
    leave_one_out = stability["leave_one_out"]
    accepted = int(recommendation["accepted"])
    selected_correct = int(recommendation["selected_correct"])
    resamples = int(leave_one_out["resamples"])
    _require_bounds("stability.full_recommendation.accepted", accepted, expected_problems)
    _require_bounds(
        "stability.full_recommendation.selected_correct",
        selected_correct,
        accepted,
    )
    for metric in ("accuracy", "coverage", "risk"):
        _require_rate_bounds(f"stability.full_recommendation.{metric}", recommendation[metric])
    _require_equal("stability.leave_one_out.resamples", resamples, expected_problems)
    _require_bounds(
        "stability.leave_one_out.threshold_changed",
        int(leave_one_out["threshold_changed"]),
        resamples,
    )
    _require_bounds(
        "stability.leave_one_out.meets_target",
        int(leave_one_out["meets_target"]),
        resamples,
    )
    _require_bounds(
        "stability.leave_one_out.unique_threshold_count",
        int(leave_one_out["unique_threshold_count"]),
        resamples,
    )
    _require_bounds(
        "stability.leave_one_out.applied_full_accepted_min",
        int(leave_one_out["applied_full_accepted_min"]),
        expected_problems,
    )
    _require_bounds(
        "stability.leave_one_out.applied_full_accepted_max",
        int(leave_one_out["applied_full_accepted_max"]),
        expected_problems,
    )
    _require_order(
        "stability.leave_one_out.applied_full_accepted",
        float(leave_one_out["applied_full_accepted_min"]),
        float(leave_one_out["applied_full_accepted_max"]),
    )
    _require_order(
        "stability.leave_one_out.threshold",
        float(leave_one_out["threshold_min"]),
        float(leave_one_out["threshold_max"]),
    )
    _require_rate_bounds(
        "stability.leave_one_out.accepted_set_jaccard_min",
        leave_one_out["accepted_set_jaccard_min"],
    )


def _validate_leaderboard_metrics(leaderboard: dict[str, Any]) -> None:
    problems = int(leaderboard["dataset"]["problems"])
    methods = leaderboard["methods"]
    best_method = leaderboard.get("best_method")
    if best_method not in methods:
        raise ValueError(
            f"Inconsistent source artifact leaderboard.best_method: "
            f"{best_method!r} not in methods"
        )
    expected_best = _leaderboard_best_method(methods)
    if best_method != expected_best:
        raise ValueError(
            f"Inconsistent source artifact leaderboard.best_method: "
            f"{best_method!r} != {expected_best!r}"
        )

    any_correct = None
    dataset_any = leaderboard["dataset"].get("has_any_correct")
    if isinstance(dataset_any, dict):
        _validate_prop(
            "leaderboard.dataset.has_any_correct",
            dataset_any,
            total=problems,
        )
        any_correct = int(dataset_any["successes"])

    for name, row in leaderboard["methods"].items():
        accepted = _leaderboard_accepted(row, problems=problems)
        abstained = int(row["abstained"])
        if "problems" in row:
            _require_equal(f"leaderboard.{name}.problems", int(row["problems"]), problems)
        _require_equal(f"leaderboard.{name}.accepted_plus_abstained", accepted + abstained, problems)
        _require_rate(f"leaderboard.{name}.coverage", row["coverage"], accepted, problems)
        _validate_prop(f"leaderboard.{name}.selected_correct", row["selected_correct"], total=problems)
        _validate_prop(f"leaderboard.{name}.accepted_accuracy", row["accepted_accuracy"], total=accepted)
        _require_equal(
            f"leaderboard.{name}.accepted_accuracy.successes",
            int(row["accepted_accuracy"]["successes"]),
            int(row["selected_correct"]["successes"]),
        )
        if "selected_correct_given_any" in row:
            _validate_prop(
                f"leaderboard.{name}.selected_correct_given_any",
                row["selected_correct_given_any"],
                successes=int(row["selected_correct"]["successes"]),
                total=any_correct,
            )
        if any_correct is not None and "missed_available_correct" in row:
            _require_equal(
                f"leaderboard.{name}.missed_available_correct",
                int(row["missed_available_correct"]),
                any_correct - int(row["selected_correct"]["successes"]),
            )

    expected_frontier = _leaderboard_frontier(methods)
    actual_frontier = leaderboard.get("coverage_accuracy_frontier", [])
    if actual_frontier != expected_frontier:
        raise ValueError(
            "Inconsistent source artifact leaderboard.coverage_accuracy_frontier"
        )

    for row in actual_frontier:
        accepted = int(row["accepted"])
        method = row["method"]
        if method not in methods:
            raise ValueError(
                f"Inconsistent source artifact frontier.{method}: method not in methods"
            )
        _require_rate(f"frontier.{row['method']}.coverage", row["coverage"], accepted, problems)
        _validate_prop(
            f"frontier.{row['method']}.accepted_accuracy",
            row["accepted_accuracy"],
            total=accepted,
        )
    _validate_leaderboard_pairwise(
        leaderboard.get("pairwise", {}),
        methods=methods,
        problems=problems,
    )


def _leaderboard_best_method(methods: dict[str, Any]) -> str:
    return sorted(
        methods,
        key=lambda name: (
            -int(methods[name]["selected_correct"]["successes"]),
            -float(methods[name]["accepted_accuracy"]["rate"]),
            name,
        ),
    )[0]


def _leaderboard_accepted(row: dict[str, Any], *, problems: int | None = None) -> int:
    if "accepted" in row:
        return int(row["accepted"])
    if problems is None:
        problems = int(row["selected_correct"]["total"])
    return int(problems) - int(row["abstained"])


def _leaderboard_frontier(methods: dict[str, Any]) -> list[dict[str, Any]]:
    frontier = []
    for name, row in methods.items():
        accepted = _leaderboard_accepted(row)
        if accepted == 0:
            continue
        coverage = float(row["coverage"])
        accuracy = float(row["accepted_accuracy"]["rate"])
        dominated = False
        for other_name, other in methods.items():
            other_accepted = _leaderboard_accepted(other)
            if other_name == name or other_accepted == 0:
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
                    "accepted": accepted,
                }
            )
    return sorted(
        frontier,
        key=lambda row: (
            -float(row["coverage"]),
            -float(row["accepted_accuracy"]["rate"]),
            row["method"],
        ),
    )


def _validate_leaderboard_pairwise(
    pairwise: dict[str, Any],
    *,
    methods: dict[str, Any],
    problems: int,
) -> None:
    method_names = set(methods)
    if not isinstance(pairwise, dict):
        raise ValueError("Inconsistent source artifact leaderboard.pairwise: missing")
    for name in sorted(method_names):
        row = pairwise.get(name)
        if not isinstance(row, dict):
            raise ValueError(
                f"Inconsistent source artifact leaderboard.pairwise.{name}: missing"
            )
        missing = sorted(method_names - {name} - set(row))
        if missing:
            shown = ", ".join(missing)
            raise ValueError(
                f"Inconsistent source artifact leaderboard.pairwise.{name}: "
                f"missing {shown}"
            )
        for other in sorted(method_names - {name}):
            _validate_pairwise_comparison(
                f"leaderboard.pairwise.{name}.{other}",
                row[other],
                total=problems,
            )


def _validate_pairwise_comparison(
    label: str,
    row: dict[str, Any],
    *,
    total: int,
) -> None:
    actual_total = int(row["total"])
    _require_equal(f"{label}.total", actual_total, total)
    both = int(row["both_success"])
    a_only = int(row["a_only"])
    b_only = int(row["b_only"])
    neither = int(row["neither_success"])
    for key, value in (
        ("both_success", both),
        ("a_only", a_only),
        ("b_only", b_only),
        ("neither_success", neither),
    ):
        _require_bounds(f"{label}.{key}", value, total)
    _require_equal(f"{label}.partition", both + a_only + b_only + neither, total)
    _require_equal(f"{label}.a_successes", int(row["a_successes"]), both + a_only)
    _require_equal(f"{label}.b_successes", int(row["b_successes"]), both + b_only)
    _require_equal(f"{label}.discordant", int(row["discordant"]), a_only + b_only)
    expected_lift = 0.0 if total == 0 else (b_only - a_only) / float(total)
    _require_close(f"{label}.b_minus_a", float(row["b_minus_a"]), expected_lift)
    _require_rate_bounds(f"{label}.exact_sign_p", row["exact_sign_p"])


def _validate_ood_metrics(ood: dict[str, Any]) -> None:
    pairs = int(ood["num_pairs"])
    positives = int(ood["num_positive"])
    negatives = int(ood["num_negative"])
    _require_equal("ood.num_positive_plus_negative", positives + negatives, pairs)
    _require_bounds("ood.num_positive", positives, pairs)
    _require_bounds("ood.num_negative", negatives, pairs)
    accuracy = float(ood["accuracy"])
    pos_accuracy = float(ood["pos_accuracy"])
    neg_accuracy = float(ood["neg_accuracy"])
    for metric in (
        "accuracy",
        "pos_accuracy",
        "neg_accuracy",
        "selection_top1_accuracy",
        "calibrated_val_accuracy",
        "calibrated_val_balanced_accuracy",
        "calibrated_test_accuracy",
        "calibrated_test_balanced_accuracy",
    ):
        if metric in ood:
            _require_rate_bounds(f"ood.{metric}", ood[metric])
    expected_accuracy = 0.0 if pairs == 0 else (
        positives * pos_accuracy + negatives * neg_accuracy
    ) / pairs
    _require_close("ood.accuracy", accuracy, expected_accuracy)
    _require_bounds(
        "ood.selection_problems",
        int(ood["selection_problems"]),
        positives,
    )
    for name, value in ood.get("per_type_accuracy", {}).items():
        _require_rate_bounds(f"ood.per_type_accuracy.{name}", value)


def _validate_prop(
    label: str,
    row: dict[str, Any],
    *,
    successes: int | None = None,
    total: int | None = None,
) -> None:
    actual_successes = int(row["successes"])
    actual_total = int(row["total"])
    if successes is not None:
        _require_equal(f"{label}.successes", actual_successes, int(successes))
    if total is not None:
        _require_equal(f"{label}.total", actual_total, int(total))
    _require_bounds(f"{label}.successes", actual_successes, actual_total)
    _require_rate(f"{label}.rate", row["rate"], actual_successes, actual_total)


def _has_any_count(row: dict[str, Any]) -> int:
    return int(
        row.get(
            "has_any_correct",
            _pct_to_count(float(row["has_any_correct_pct"]), int(row["problems"])),
        )
    )


def _require_equal(label: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError(f"Inconsistent source artifact {label}: {actual} != {expected}")


def _require_bounds(label: str, value: int, total: int) -> None:
    if value < 0 or value > total:
        raise ValueError(f"Inconsistent source artifact {label}: {value} outside [0, {total}]")


def _require_pct_count(label: str, pct: Any, count: int, total: int) -> None:
    _require_rate(label, float(pct) / 100.0, count, total)


def _require_rate(label: str, actual: Any, successes: int, total: int) -> None:
    expected = 0.0 if total == 0 else float(successes) / float(total)
    _require_rate_bounds(label, actual)
    _require_rate_bounds(f"{label}.expected", expected)
    _require_close(label, float(actual), expected)


def _require_rate_bounds(label: str, value: Any) -> None:
    numeric = float(value)
    if numeric < -1e-12 or numeric > 1.0 + 1e-12:
        raise ValueError(
            f"Inconsistent source artifact {label}: {numeric} outside [0, 1]"
        )


def _require_order(label: str, lower: float, upper: float) -> None:
    if lower > upper:
        raise ValueError(f"Inconsistent source artifact {label}: {lower} > {upper}")


def _require_close(label: str, actual: float, expected: float) -> None:
    if abs(float(actual) - expected) > 1e-9:
        raise ValueError(
            f"Inconsistent source artifact {label}: {float(actual)} != {expected}"
        )


def _split_meta(rows: dict[str, Any]) -> dict[str, int]:
    first = next(iter(rows.values()))
    problems = int(first["problems"])
    has_any = _pct_to_count(float(first["has_any_correct_pct"]), problems)
    return {"problems": problems, "has_any_correct": has_any}


def _main_table_metric(row: dict[str, Any]) -> dict[str, Any]:
    problems = int(row["problems"])
    has_any = int(
        row.get(
            "has_any_correct",
            _pct_to_count(float(row["has_any_correct_pct"]), problems),
        )
    )
    return {
        "problems": problems,
        "has_any_correct": has_any,
        "selected_correct": {
            "successes": _pct_to_count(float(row["selected_correct_pct"]), problems),
            "total": problems,
            "pct": float(row["selected_correct_pct"]),
        },
        "selected_correct_given_any": {
            "successes": _pct_to_count(
                float(row["selected_correct_given_any_pct"]),
                has_any,
            ),
            "total": has_any,
            "pct": float(row["selected_correct_given_any_pct"]),
        },
    }


def _append_proofnetverif_table(
    lines: list[str],
    *,
    title: str,
    split: str,
    summary: dict[str, Any],
) -> None:
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            "| setting | selected_correct (%) | selected_correct_given_any (%) | any_correct (%) | MRR | Hit@3 (%) | Hit@5 (%) | problems |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in SPLITS[split]:
        row = summary["proofnetverif"][split][key]
        lines.append(
            f"| {label} | {row['selected_correct_pct']:.1f} | "
            f"{row['selected_correct_given_any_pct']:.1f} | "
            f"{row['has_any_correct_pct']:.1f} | {row['mrr']:.3f} | "
            f"{row['hit_at_3']:.1f} | {row['hit_at_5']:.1f} | "
            f"{int(row['problems'])} |"
        )


def _append_selective_prediction(
    lines: list[str],
    selective: dict[str, Any],
) -> None:
    confidence = selective["confidence_signals"]
    abstention = selective["abstention_policy_p50"]
    stability = abstention["threshold_stability"]
    lines.extend(
        [
            "",
            "## Selective prediction and abstention (inductive)",
            "",
            f"Best integrated confidence signal: "
            f"`{confidence['best_by_oracle_normalized_accuracy_area']}`.",
            "",
            "| confidence key | mean prefix accuracy | lift over full | average precision | oracle-normalized area | best prefix accuracy | best prefix coverage |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, row in confidence["signals"].items():
        lines.append(
            f"| `{key}` | {_pct(row['mean_prefix_accuracy'])} | "
            f"{_pp(row['accuracy_lift_over_full'])} | "
            f"{_pct(row['average_precision'])} | "
            f"{_pct(row['oracle_normalized_accuracy_area'])} | "
            f"{_pct(row['best_prefix_accuracy'])} | "
            f"{_pct(row['best_prefix_coverage'])} |"
        )

    lines.extend(
        [
            "",
            "Certified 50% Wilson-LCB abstention policy:",
            "",
            "| coverage | accepted accuracy | selected correct counting abstentions | accepted selected-correct given any | accepted | abstained | threshold |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {_pct(abstention['coverage']['rate'])} | "
            f"{_prop(abstention['accepted_selected_correct'])} | "
            f"{_prop(abstention['selected_correct_counting_abstentions_incorrect'])} | "
            f"{_prop(abstention['accepted_selected_correct_given_any'])} | "
            f"{abstention['accepted']} | {abstention['abstained']} | "
            f"{abstention['threshold']:.4f} |",
            "",
            "Leave-one-out threshold stability:",
            "",
            f"- Unique thresholds: `{stability['unique_threshold_count']}`",
            f"- Threshold range: `{stability['threshold_min']:.4f}` to "
            f"`{stability['threshold_max']:.4f}`",
            f"- Changed threshold in `{stability['threshold_changed']}/"
            f"{stability['resamples']}` folds",
            f"- Applied full-sample accepted range: "
            f"`{stability['applied_full_accepted_min']}` to "
            f"`{stability['applied_full_accepted_max']}`",
            f"- Minimum accepted-set Jaccard: "
            f"`{100.0 * stability['accepted_set_jaccard_min']:.1f}%`",
            "",
            "Coverage/accuracy frontier:",
            "",
            "| method | coverage | accepted accuracy | accepted |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in abstention["coverage_accuracy_frontier"]:
        lines.append(
            f"| `{row['method']}` | {_pct(row['coverage'])} | "
            f"{_prop(row['accepted_accuracy'])} | {row['accepted']} |"
        )


def _append_leaderboard(lines: list[str], leaderboard: dict[str, Any]) -> None:
    lines.extend(
        [
            "",
            "## Multi-method selection leaderboard (inductive)",
            "",
            f"Best method by selected-correct count: `{leaderboard['best_method']}`.",
            "",
            "| method | coverage | selected correct | accepted accuracy | missed available | abstained |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in leaderboard["methods"].items():
        lines.append(
            f"| `{name}` | {_pct(row['coverage'])} | "
            f"{_prop(row['selected_correct'])} | {_prop(row['accepted_accuracy'])} | "
            f"{row['missed_available_correct']} | {row['abstained']} |"
        )


def _append_ood(lines: list[str], ood: dict[str, Any]) -> None:
    lines.extend(
        [
            "",
            "## OOD pair classification (FormalAlign minif2f misalignment test)",
            "",
            f"Pairs: {ood['num_pairs']} (pos={ood['num_positive']}, neg={ood['num_negative']})",
            "Zero-shot accuracy @thr=0: "
            f"{_pct(ood['accuracy'])} (pos={_pct(ood['pos_accuracy'])}, "
            f"neg={_pct(ood['neg_accuracy'])})",
            "Selection top1 accuracy (per-input): "
            f"{_pct2(ood['selection_top1_accuracy'])} "
            f"(n={ood['selection_problems']})",
            "",
            "| misalignment type | accuracy |",
            "|---|---:|",
        ]
    )
    for name, value in ood.get("per_type_accuracy", {}).items():
        lines.append(f"| `{name}` | {_pct(value)} |")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(Path.cwd())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _file_sha256(path: Path) -> str:
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _pct2(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _pct_to_count(pct: float, total: int) -> int:
    return int(round((float(pct) / 100.0) * int(total)))


def _pp(value: float) -> str:
    return f"{100.0 * float(value):+.1f} pp"


def _prop(row: dict[str, Any]) -> str:
    return f"{_pct(row['rate'])} ({row['successes']}/{row['total']})"


def _latex_prop(row: dict[str, Any], *, bold: bool = False) -> str:
    text = f"{row['successes']}/{row['total']} ({row['pct']:.1f}\\%)"
    return rf"\textbf{{{text}}}" if bold else text


def _stale_outputs(expected: dict[Path, str]) -> list[Path]:
    stale = []
    for path, expected_text in expected.items():
        if not path.exists() or path.read_text(encoding="utf-8") != expected_text:
            stale.append(path)
    return stale


def _source_hash_mismatches(
    summary: dict[str, Any],
    *,
    base_dir: Path,
) -> list[dict[str, str]]:
    provenance = summary.get("provenance", {})
    sources = provenance.get("sources", {})
    expected_hashes = provenance.get("source_sha256", {})
    mismatches = []
    for name, path_text in sources.items():
        expected = expected_hashes.get(name)
        if not expected:
            mismatches.append(
                {
                    "name": str(name),
                    "path": str(path_text),
                    "expected": "<missing from source_sha256>",
                    "actual": "<not checked>",
                }
            )
            continue
        path = _resolve_source_path(str(path_text), base_dir=base_dir)
        if not path.exists():
            mismatches.append(
                {
                    "name": str(name),
                    "path": str(path_text),
                    "expected": expected,
                    "actual": "<missing file>",
                }
            )
            continue
        actual = _file_sha256(path)
        if actual != expected:
            mismatches.append(
                {
                    "name": str(name),
                    "path": str(path_text),
                    "expected": expected,
                    "actual": actual,
                }
            )
    for name in sorted(set(expected_hashes) - set(sources)):
        mismatches.append(
            {
                "name": str(name),
                "path": "<missing from sources>",
                "expected": str(expected_hashes[name]),
                "actual": "<not checked>",
            }
        )
    return mismatches


def _resolve_source_path(path_text: str, *, base_dir: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else base_dir / path


def _display_paths(paths: list[Path]) -> str:
    return ", ".join(str(path).replace("\\", "/") for path in paths)


def _display_hash_mismatches(mismatches: list[dict[str, str]]) -> str:
    return "; ".join(
        f"{row['name']} {row['path']} expected {row['expected']} got {row['actual']}"
        for row in mismatches
    )


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--ablation-json", default="")
    parser.add_argument("--output-json", default="results/nlverifier_paper_metrics.json")
    parser.add_argument("--output-md", default="results/nlverifier_paper_metrics.md")
    parser.add_argument("--output-tex", default="paper/generated/nlverifier_main_table.tex")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if generated paper artifacts differ from the checked-in files",
    )
    parser.add_argument(
        "--verify-source-hashes",
        action="store_true",
        help="validate source_sha256 entries in --output-json against current source files",
    )
    args = parser.parse_args()

    output_json = Path(args.output_json)
    if args.verify_source_hashes:
        summary = _load_json(output_json)
        mismatches = _source_hash_mismatches(summary, base_dir=Path.cwd())
        if mismatches:
            raise SystemExit(
                "Source hash verification failed for "
                + str(output_json).replace("\\", "/")
                + ": "
                + _display_hash_mismatches(mismatches)
            )
        print(f"Source hashes verified for {output_json}")
        return

    summary = build_summary(
        Path(args.results_dir),
        ablation_json=Path(args.ablation_json) if args.ablation_json else None,
    )
    json_text = json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    markdown = format_markdown(summary)
    latex = format_latex_main_table(summary)
    output_md = Path(args.output_md)
    output_tex = Path(args.output_tex)
    expected = {
        output_json: json_text,
        output_md: markdown,
        output_tex: latex,
    }

    if args.check:
        stale = _stale_outputs(expected)
        if stale:
            raise SystemExit(
                "Stale NLVerifier paper artifacts: "
                + _display_paths(stale)
                + "; rerun scripts/summarize_nlverifier_paper_metrics.py without --check"
            )
        print(f"NLVerifier paper artifacts are up to date: {_display_paths(list(expected))}")
        return

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    _write_text(output_json, json_text)
    _write_text(output_md, markdown)
    _write_text(output_tex, latex)
    print(f"Wrote {output_json}, {output_md}, and {output_tex}")


if __name__ == "__main__":
    main()
