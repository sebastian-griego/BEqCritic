#!/usr/bin/env python3
"""Generate paper-ready NLVerifier metrics from checked-in result artifacts."""

from __future__ import annotations

import argparse
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

    ablation = _load_json(ablation_path)
    confidence = _load_json(confidence_path)
    abstention = _load_json(abstention_path)
    stability = _load_json(stability_path)
    leaderboard = _load_json(leaderboard_path)
    ood = _load_json(ood_path)

    settings = dict(ablation["settings"])
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

    return {
        **settings,
        "provenance": {
            **dict(ablation.get("provenance", {})),
            "generator": "scripts/summarize_nlverifier_paper_metrics.py",
            "sources": {
                "proofnetverif_ablation": _rel(ablation_path),
                "confidence_audit": _rel(confidence_path),
                "abstention_metrics": _rel(abstention_path),
                "threshold_stability": _rel(stability_path),
                "selection_leaderboard": _rel(leaderboard_path),
                "ood_formalalign": _rel(ood_path),
            },
        },
        "proofnetverif": proofnetverif,
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
    for name, path in provenance["sources"].items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines).rstrip() + "\n"


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


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _pct2(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _pp(value: float) -> str:
    return f"{100.0 * float(value):+.1f} pp"


def _prop(row: dict[str, Any]) -> str:
    return f"{_pct(row['rate'])} ({row['successes']}/{row['total']})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--ablation-json", default="")
    parser.add_argument("--output-json", default="results/nlverifier_paper_metrics.json")
    parser.add_argument("--output-md", default="results/nlverifier_paper_metrics.md")
    args = parser.parse_args()

    summary = build_summary(
        Path(args.results_dir),
        ablation_json=Path(args.ablation_json) if args.ablation_json else None,
    )
    markdown = format_markdown(summary)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown, encoding="utf-8")
    print(f"Wrote {output_json} and {output_md}")


if __name__ == "__main__":
    main()
