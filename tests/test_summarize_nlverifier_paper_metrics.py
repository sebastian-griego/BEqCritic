from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.summarize_nlverifier_paper_metrics import (
    _source_hash_mismatches,
    _stale_outputs,
    build_summary,
    format_latex_main_table,
    format_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_nlverifier_paper_metrics_summary_merges_source_artifacts(tmp_path):
    results = tmp_path / "results"
    _write_summary_inputs(results)

    summary = build_summary(results)
    markdown = format_markdown(summary)
    latex = format_latex_main_table(summary)

    assert summary["transductive_nl"]["selected_correct_pct"] == 60.0
    assert set(summary["provenance"]["source_sha256"]) == set(summary["provenance"]["sources"])
    assert all(
        len(digest) == 64
        for digest in summary["provenance"]["source_sha256"].values()
    )
    assert _source_hash_mismatches(summary, base_dir=ROOT) == []
    assert summary["proofnetverif"]["inductive"]["inductive_nl"]["problems"] == 5
    assert summary["main_table"]["rows"][2]["key"] == "critic_global_medoid"
    assert summary["main_table"]["rows"][2]["transductive"]["selected_correct"]["successes"] == 6
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
    assert "sha256:" in markdown
    assert "Candidate-only critic (best variant)" in latex
    assert r"\resizebox{\textwidth}{!}{%" in latex
    assert r"\textbf{3/5 (60.0\%)}" in latex


def test_nlverifier_paper_metrics_rejects_inconsistent_denominators(tmp_path):
    results = tmp_path / "results"
    _write_summary_inputs(results)
    _write_json(
        results / "exp_inductive" / "metrics_random.json",
        _metric("random", 6, 4, 1),
    )

    with pytest.raises(ValueError, match="inductive_random.problems"):
        build_summary(results)


def test_nlverifier_paper_metrics_rejects_inconsistent_percentages(tmp_path):
    results = tmp_path / "results"
    _write_summary_inputs(results)
    metric = _metric("random", 5, 4, 1)
    metric["selected_correct_pct"] = 50.0
    _write_json(results / "exp_inductive" / "metrics_random.json", metric)

    with pytest.raises(ValueError, match="inductive_random.selected_correct"):
        build_summary(results)


def test_paper_metrics_check_cli_fails_without_rewriting_stale_outputs(tmp_path):
    results = tmp_path / "results"
    _write_summary_inputs(results)
    output_json = tmp_path / "paper_metrics.json"
    output_md = tmp_path / "paper_metrics.md"
    output_tex = tmp_path / "main_table.tex"
    for output in (output_json, output_md, output_tex):
        output.write_text("stale\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_nlverifier_paper_metrics.py"),
            "--results-dir",
            str(results),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--output-tex",
            str(output_tex),
            "--check",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Stale NLVerifier paper artifacts" in completed.stderr + completed.stdout
    assert output_json.read_text(encoding="utf-8") == "stale\n"
    assert output_md.read_text(encoding="utf-8") == "stale\n"
    assert output_tex.read_text(encoding="utf-8") == "stale\n"


def test_paper_metrics_verify_source_hashes_cli_detects_changed_source(tmp_path):
    results = tmp_path / "results"
    _write_summary_inputs(results)
    output_json = tmp_path / "paper_metrics.json"
    output_md = tmp_path / "paper_metrics.md"
    output_tex = tmp_path / "main_table.tex"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_nlverifier_paper_metrics.py"),
            "--results-dir",
            str(results),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--output-tex",
            str(output_tex),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_nlverifier_paper_metrics.py"),
            "--output-json",
            str(output_json),
            "--verify-source-hashes",
        ],
        cwd=ROOT,
        check=True,
    )

    _write_json(results / "ood_formalalign_minif2f.json", {**_ood(), "num_pairs": 13})
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_nlverifier_paper_metrics.py"),
            "--output-json",
            str(output_json),
            "--verify-source-hashes",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Source hash verification failed" in completed.stderr + completed.stdout
    assert "ood_formalalign" in completed.stderr + completed.stdout


def test_stale_outputs_detects_missing_and_mismatched_files(tmp_path):
    matching = tmp_path / "matching.txt"
    stale = tmp_path / "stale.txt"
    missing = tmp_path / "missing.txt"
    matching.write_text("expected\n", encoding="utf-8")
    stale.write_text("old\n", encoding="utf-8")

    assert _stale_outputs(
        {
            matching: "expected\n",
            stale: "new\n",
            missing: "new\n",
        }
    ) == [stale, missing]


def test_checked_in_nlverifier_paper_artifacts_are_current(monkeypatch):
    monkeypatch.chdir(ROOT)
    summary = build_summary(Path("results"))

    assert Path("results/nlverifier_paper_metrics.json").read_text(encoding="utf-8") == (
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    )
    assert Path("results/nlverifier_paper_metrics.md").read_text(
        encoding="utf-8"
    ) == format_markdown(summary)
    assert Path("paper/generated/nlverifier_main_table.tex").read_text(
        encoding="utf-8"
    ) == format_latex_main_table(summary)


def _write_summary_inputs(results: Path) -> None:
    _write_json(results / "nlverifier_proofnetverif_ablation_metrics.json", _ablation())
    _write_json(results / "exp_transductive" / "metrics_random.json", _metric("random", 10, 8, 4))
    _write_json(
        results / "exp_transductive" / "metrics_self_bleu.json",
        _metric("self_bleu", 10, 8, 5),
    )
    _write_json(
        results / "exp_transductive" / "metrics_critic_global_medoid.json",
        _metric("critic_global_medoid", 10, 8, 6),
    )
    _write_json(results / "exp_inductive" / "metrics_random.json", _metric("random", 5, 4, 1))
    _write_json(
        results / "exp_inductive" / "metrics_self_bleu.json",
        _metric("self_bleu", 5, 4, 2),
    )
    _write_json(
        results / "exp_inductive" / "metrics_critic_global_medoid.json",
        _metric("critic_global_medoid", 5, 4, 2),
    )
    _write_json(results / "exp_inductive" / "nlverifier_confidence_audit.json", _confidence())
    _write_json(results / "exp_inductive" / "metrics_nlverifier_abstain_p50.json", _abstention())
    _write_json(
        results / "exp_inductive" / "nlverifier_threshold_stability_p50.json",
        _stability(),
    )
    _write_json(results / "exp_inductive" / "selection_leaderboard.json", _leaderboard())
    _write_json(results / "ood_formalalign_minif2f.json", _ood())


def _metric(name: str, problems: int, has_any: int, selected: int) -> dict:
    return {
        "avg_candidates_per_problem": 2.0,
        "has_any_correct": has_any,
        "has_any_correct_pct": 100.0 * has_any / problems,
        "name": name,
        "problems": problems,
        "selected_correct_given_any_pct": 100.0 * selected / has_any,
        "selected_correct_pct": 100.0 * selected / problems,
    }


def _setting(problems: int, selected: float, given_any: float) -> dict:
    return {
        "problems": problems,
        "avg_candidates_per_problem": 2.0,
        "selected_correct_pct": selected,
        "has_any_correct_pct": 80.0,
        "selected_correct_given_any_pct": given_any,
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
            "transductive_nl": _setting(10, 60.0, 75.0),
            "transductive_nl_blank": _setting(10, 50.0, 62.5),
            "transductive_nl_const": _setting(10, 60.0, 75.0),
            "inductive_nl": _setting(5, 60.0, 75.0),
            "inductive_nl_blank": _setting(5, 40.0, 50.0),
            "inductive_nl_const": _setting(5, 40.0, 50.0),
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
