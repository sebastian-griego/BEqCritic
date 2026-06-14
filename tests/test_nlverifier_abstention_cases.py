from __future__ import annotations

import json
import subprocess
import sys
from math import isclose

from beqcritic.nlverifier_abstention_cases import analyze_cases, format_markdown


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_analyze_cases_groups_abstention_outcomes(tmp_path):
    scores = tmp_path / "scores.jsonl"
    accepted = tmp_path / "accepted.jsonl"
    abstained = tmp_path / "abstained.jsonl"
    _write_jsonl(
        scores,
        [
            {
                "problem_id": "p1",
                "candidates": ["wrong", "right"],
                "labels": [0, 1],
                "scores": [3.0, 2.0],
            },
            {
                "problem_id": "p2",
                "candidates": ["right", "wrong"],
                "labels": [1, 0],
                "scores": [0.3, 0.2],
            },
            {
                "problem_id": "p3",
                "candidates": ["wrong", "right"],
                "labels": [0, 1],
                "scores": [0.4, 0.9],
            },
            {
                "problem_id": "p4",
                "candidates": ["wrong", "also wrong"],
                "labels": [0, 0],
                "scores": [5.0, 4.0],
            },
        ],
    )
    _write_jsonl(
        accepted,
        [
            {
                "problem_id": "p1",
                "chosen_index": 0,
                "accepted": True,
                "confidence_key": "chosen_probability",
                "confidence": 0.9,
                "threshold": 0.5,
            },
            {
                "problem_id": "p4",
                "chosen_index": 0,
                "accepted": True,
                "confidence_key": "chosen_probability",
                "confidence": 0.7,
                "threshold": 0.5,
            },
        ],
    )
    _write_jsonl(
        abstained,
        [
            {
                "problem_id": "p2",
                "chosen_index": 0,
                "abstained": True,
                "confidence_key": "chosen_probability",
                "confidence": 0.4,
                "threshold": 0.5,
            },
            {
                "problem_id": "p3",
                "chosen_index": 0,
                "abstained": True,
                "confidence_key": "chosen_probability",
                "confidence": 0.3,
                "threshold": 0.5,
            },
        ],
    )

    summary = analyze_cases(
        scores_path=scores,
        accepted_path=accepted,
        abstained_path=abstained,
        top_candidates=2,
    )
    text = format_markdown(summary)

    assert summary["dataset"]["problems"] == 4
    assert summary["dataset"]["accepted_errors"] == 2
    assert summary["dataset"]["accepted_missed_available_correct"] == 1
    assert summary["dataset"]["accepted_no_correct_candidate"] == 1
    assert summary["dataset"]["abstained_selected_correct"] == 1
    assert summary["dataset"]["abstained_missed_available_correct"] == 1
    assert summary["case_counts"]["accepted_missed_available_correct"] == 1
    accepted_miss = summary["casebook"]["accepted_missed_available_correct"][0]
    assert accepted_miss["problem_id"] == "p1"
    assert accepted_miss["best_correct_index"] == 1
    assert accepted_miss["best_correct_rank"] == 2
    assert accepted_miss["score_gap_to_best_correct"] == 1.0
    abstained_correct = summary["casebook"]["abstained_correct_selections"][0]
    assert abstained_correct["problem_id"] == "p2"
    assert isclose(abstained_correct["threshold_distance"], -0.1, abs_tol=1e-12)
    assert "NLVerifier Abstention Casebook" in text
    assert "Accepted Misses With Correct Candidate Available" in text


def test_nlverifier_abstention_cases_cli_writes_outputs(tmp_path):
    scores = tmp_path / "scores.jsonl"
    accepted = tmp_path / "accepted.jsonl"
    abstained = tmp_path / "abstained.jsonl"
    output_json = tmp_path / "cases.json"
    output_md = tmp_path / "cases.md"
    cases_jsonl = tmp_path / "cases.jsonl"
    _write_jsonl(
        scores,
        [
            {
                "problem_id": "p1",
                "candidates": ["wrong", "right"],
                "labels": [0, 1],
                "scores": [3.0, 2.0],
            },
            {
                "problem_id": "p2",
                "candidates": ["right", "wrong"],
                "labels": [1, 0],
                "scores": [0.3, 0.2],
            },
        ],
    )
    _write_jsonl(
        accepted,
        [
            {
                "problem_id": "p1",
                "chosen_index": 0,
                "accepted": True,
                "confidence_key": "chosen_probability",
                "confidence": 0.9,
                "threshold": 0.5,
            }
        ],
    )
    _write_jsonl(
        abstained,
        [
            {
                "problem_id": "p2",
                "chosen_index": 0,
                "abstained": True,
                "confidence_key": "chosen_probability",
                "confidence": 0.4,
                "threshold": 0.5,
            }
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.nlverifier_abstention_cases",
            "--scores",
            str(scores),
            "--accepted",
            str(accepted),
            "--abstained",
            str(abstained),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--cases-jsonl",
            str(cases_jsonl),
        ],
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    case_rows = [
        json.loads(line)
        for line in cases_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert payload["dataset"]["accepted_missed_available_correct"] == 1
    assert payload["dataset"]["abstained_selected_correct"] == 1
    assert len(case_rows) == 2
    assert "Case Counts" in output_md.read_text(encoding="utf-8")
