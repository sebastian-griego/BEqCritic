from __future__ import annotations

import json

from scripts.summarize_nlverifier_paper_metrics import build_summary, format_markdown


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_nlverifier_paper_metrics_summary_merges_source_artifacts(tmp_path):
    results = tmp_path / "results"
    _write_json(results / "nlverifier_proofnetverif_ablation_metrics.json", _ablation())
    _write_json(results / "exp_inductive" / "nlverifier_confidence_audit.json", _confidence())
    _write_json(results / "exp_inductive" / "metrics_nlverifier_abstain_p50.json", _abstention())
    _write_json(
        results / "exp_inductive" / "nlverifier_threshold_stability_p50.json",
        _stability(),
    )
    _write_json(results / "exp_inductive" / "selection_leaderboard.json", _leaderboard())
    _write_json(results / "ood_formalalign_minif2f.json", _ood())

    summary = build_summary(results)
    markdown = format_markdown(summary)

    assert summary["transductive_nl"]["selected_correct_pct"] == 62.0
    assert summary["proofnetverif"]["inductive"]["inductive_nl"]["problems"] == 5
    assert (
        summary["selective_prediction"]["confidence_signals"][
            "best_by_oracle_normalized_accuracy_area"
        ]
        == "chosen_probability"
    )
    assert summary["selective_prediction"]["abstention_policy_p50"]["accepted"] == 3
    assert summary["leaderboard"]["best_method"] == "nlverifier"
    assert summary["ood_formalalign_minif2f"]["num_pairs"] == 12
    assert "Selective prediction and abstention" in markdown
    assert "OOD pair classification" in markdown


def _setting(problems: int, selected: float) -> dict:
    return {
        "problems": problems,
        "avg_candidates_per_problem": 2.0,
        "selected_correct_pct": selected,
        "has_any_correct_pct": 80.0,
        "selected_correct_given_any_pct": 75.0,
        "mrr": 0.8,
        "hit_at_3": 90.0,
        "hit_at_5": 100.0,
    }


def _ablation() -> dict:
    return {
        "provenance": {
            "model": "model",
            "nl_blank": "empty string",
            "nl_const": "constant",
        },
        "settings": {
            "transductive_nl": _setting(10, 62.0),
            "transductive_nl_blank": _setting(10, 55.0),
            "transductive_nl_const": _setting(10, 58.0),
            "inductive_nl": _setting(5, 50.0),
            "inductive_nl_blank": _setting(5, 40.0),
            "inductive_nl_const": _setting(5, 45.0),
        },
    }


def _confidence() -> dict:
    return {
        "dataset": {
            "problems": 5,
            "selected_correct": 3,
            "selected_accuracy": 0.6,
            "has_any_correct": 4,
            "has_any_correct_rate": 0.8,
        },
        "best_by_mean_prefix_risk": "chosen_probability",
        "best_by_oracle_normalized_accuracy_area": "chosen_probability",
        "signals": {
            "chosen_probability": {
                "mean_prefix_accuracy": 0.7,
                "mean_prefix_risk": 0.3,
                "best_accuracy_prefix": {"accuracy": 0.8, "coverage": 0.4},
                "ranking_metrics": {
                    "area_under_accuracy_coverage": 0.7,
                    "accuracy_lift_over_full": 0.1,
                    "average_precision": 0.72,
                    "oracle_normalized_accuracy_area": 0.8,
                },
            },
            "score_margin": {
                "mean_prefix_accuracy": 0.62,
                "mean_prefix_risk": 0.38,
                "best_accuracy_prefix": {"accuracy": 1.0, "coverage": 0.2},
                "ranking_metrics": {
                    "area_under_accuracy_coverage": 0.62,
                    "accuracy_lift_over_full": 0.02,
                    "average_precision": 0.61,
                    "oracle_normalized_accuracy_area": 0.55,
                },
            },
        },
    }


def _prop(successes: int, total: int) -> dict:
    return {"rate": successes / total, "successes": successes, "total": total}


def _abstention() -> dict:
    return {
        "accepted": 3,
        "abstained": 2,
        "coverage": _prop(3, 5),
        "accepted_selected_correct": _prop(2, 3),
        "accepted_selected_correct_given_any": _prop(2, 2),
        "selected_correct_counting_abstentions_incorrect": _prop(2, 5),
        "selected_correct_with_abstention_choices": _prop(3, 5),
        "accepted_has_any_correct": _prop(2, 3),
    }


def _stability() -> dict:
    return {
        "policy": {"confidence_key": "chosen_probability", "target_accuracy": 0.5},
        "full_recommendation": {"threshold": 0.7},
        "leave_one_out": {
            "unique_threshold_count": 2,
            "threshold_min": 0.6,
            "threshold_max": 0.8,
            "threshold_changed": 3,
            "resamples": 5,
            "applied_full_accepted_min": 2,
            "applied_full_accepted_max": 4,
            "accepted_set_jaccard_min": 0.75,
        },
    }


def _leaderboard() -> dict:
    return {
        "best_method": "nlverifier",
        "dataset": {"problems": 5},
        "coverage_accuracy_frontier": [
            {
                "method": "nlverifier",
                "coverage": 1.0,
                "accepted": 5,
                "accepted_accuracy": _prop(3, 5),
            }
        ],
        "methods": {
            "nlverifier": {
                "coverage": 1.0,
                "selected_correct": _prop(3, 5),
                "accepted_accuracy": _prop(3, 5),
                "missed_available_correct": 1,
                "abstained": 0,
            },
            "self_bleu": {
                "coverage": 1.0,
                "selected_correct": _prop(2, 5),
                "accepted_accuracy": _prop(2, 5),
                "missed_available_correct": 2,
                "abstained": 0,
            },
        },
        "pairwise": {
            "nlverifier": {
                "self_bleu": {"best_only": 1, "compared_only": 0}
            }
        },
    }


def _ood() -> dict:
    return {
        "dataset": "FormalAlign",
        "num_pairs": 12,
        "num_positive": 3,
        "num_negative": 9,
        "accuracy": 0.75,
        "pos_accuracy": 0.33,
        "neg_accuracy": 0.89,
        "selection_top1_accuracy": 0.25,
        "selection_problems": 3,
        "per_type_accuracy": {"constant": 0.8},
    }
