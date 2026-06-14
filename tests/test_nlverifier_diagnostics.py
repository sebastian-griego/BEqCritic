from __future__ import annotations

import json
import subprocess
import sys
from math import isclose

from beqcritic.nlverifier_diagnostics import analyze_scores, format_markdown


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_analyze_scores_reports_ranking_and_failures(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections = tmp_path / "selections.jsonl"

    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["a", "b"], "labels": [1, 0]},
            {"problem_id": "p2", "candidates": ["a", "b", "c"], "labels": [0, 1, 0]},
            {"problem_id": "p3", "candidates": ["a", "b"], "labels": [0, 0]},
        ],
    )
    _write_jsonl(
        selections,
        [
            {"problem_id": "p1", "chosen_index": 0, "scores": [0.9, 0.1]},
            {"problem_id": "p2", "chosen_index": 0, "scores": [0.8, 0.7, 0.2]},
            {"problem_id": "p3", "chosen_index": 0, "scores": [0.4, 0.3]},
        ],
    )

    summary = analyze_scores(
        candidates_path=candidates,
        selections_path=selections,
        top_ks=[1, 2],
        failure_top_k=5,
    )

    assert summary["dataset"]["problems"] == 3
    assert summary["dataset"]["has_any_correct"]["successes"] == 2
    assert summary["selection"]["selected_correct"]["successes"] == 1
    assert summary["selection"]["selected_correct_given_any"]["successes"] == 1
    assert summary["ranking"]["top_k_hit"]["1"]["successes"] == 1
    assert summary["ranking"]["top_k_hit"]["2"]["successes"] == 2
    assert isclose(summary["ranking"]["mean_reciprocal_rank"], 0.5, abs_tol=1e-12)
    assert isclose(summary["ranking"]["mean_within_problem_pair_auc"], 0.75, abs_tol=1e-12)
    assert isclose(summary["ranking"]["pooled_candidate_auc"], 0.9, abs_tol=1e-12)
    assert summary["failures"]["reachable_misses"] == 1
    assert summary["failures"]["top"][0]["problem_id"] == "p2"
    assert summary["failures"]["top"][0]["best_correct_rank"] == 2
    assert "NLVerifier score diagnostics" in format_markdown(summary)


def test_nlverifier_diagnostics_cli_writes_outputs(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections = tmp_path / "selections.jsonl"
    summary_json = tmp_path / "summary.json"
    summary_md = tmp_path / "summary.md"
    failures = tmp_path / "failures.jsonl"

    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["a", "b"], "labels": [0, 1]},
        ],
    )
    _write_jsonl(
        selections,
        [
            {"problem_id": "p1", "chosen_index": 0, "scores": [2.0, 1.0]},
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.nlverifier_diagnostics",
            "--candidates",
            str(candidates),
            "--selections",
            str(selections),
            "--top-ks",
            "1,2",
            "--output-json",
            str(summary_json),
            "--output-md",
            str(summary_md),
            "--failures-jsonl",
            str(failures),
        ],
        check=True,
    )

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["failures"]["reachable_misses"] == 1
    assert "Reachable Misses" in summary_md.read_text(encoding="utf-8")
    lines = failures.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["best_correct_index"] == 1
