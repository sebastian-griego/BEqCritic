from __future__ import annotations

import json
import subprocess
import sys
from math import isclose

import pytest

from beqcritic.nlverifier_selective import (
    analyze_selective_risk,
    load_selective_examples,
)


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_selective_risk_orders_by_probability_margin(tmp_path):
    scores = tmp_path / "scores.jsonl"
    _write_jsonl(
        scores,
        [
            {
                "problem_id": "p1",
                "candidates": ["a", "b"],
                "labels": [1, 0],
                "scores": [4.0, -2.0],
                "chosen_index": 0,
            },
            {
                "problem_id": "p2",
                "candidates": ["a", "b"],
                "labels": [0, 1],
                "scores": [3.0, 2.8],
                "chosen_index": 0,
            },
            {
                "problem_id": "p3",
                "candidates": ["a", "b"],
                "labels": [0, 1],
                "scores": [0.2, 0.1],
                "chosen_index": 0,
            },
        ],
    )

    examples = load_selective_examples(scored_path=scores, temperature=2.0)
    summary = analyze_selective_risk(
        examples,
        confidence_key="probability_margin",
        min_coverages=[1 / 3, 2 / 3, 1.0],
    )

    assert summary["dataset"]["problems"] == 3
    assert summary["dataset"]["selected_correct"] == 1
    assert summary["coverage_table"][0]["accepted"] == 1
    assert summary["coverage_table"][0]["selected_correct"] == 1
    assert isclose(summary["coverage_table"][0]["accuracy"], 1.0)
    assert summary["coverage_table"][-1]["accepted"] == 3
    assert summary["high_confidence_errors"][0]["problem_id"] == "p2"


def test_selective_examples_can_join_candidates_and_selections(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections = tmp_path / "selections.jsonl"
    calibration = tmp_path / "calibration.json"
    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["a", "b"], "labels": [0, 1]},
        ],
    )
    _write_jsonl(
        selections,
        [
            {"problem_id": "p1", "chosen_index": 1, "scores": [-1.0, 2.0]},
        ],
    )
    calibration.write_text(
        json.dumps({"temperature": {"fitted": 2.0}}),
        encoding="utf-8",
    )

    examples = load_selective_examples(
        candidates_path=candidates,
        selections_path=selections,
        calibration_json=calibration,
    )

    assert len(examples) == 1
    assert examples[0].selected_correct is True
    assert examples[0].chosen_index == 1
    assert isclose(examples[0].chosen_probability, 0.7310585786, abs_tol=1e-9)


def test_selective_examples_reject_unmatched_candidate_selection_ids(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections = tmp_path / "selections.jsonl"
    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["a", "b"], "labels": [0, 1]},
            {"problem_id": "candidate_only", "candidates": ["a"], "labels": [1]},
        ],
    )
    _write_jsonl(
        selections,
        [
            {"problem_id": "p1", "chosen_index": 1, "scores": [-1.0, 2.0]},
            {"problem_id": "selection_only", "chosen_index": 0, "scores": [1.0]},
        ],
    )

    with pytest.raises(ValueError) as excinfo:
        load_selective_examples(
            candidates_path=candidates,
            selections_path=selections,
        )

    message = str(excinfo.value)
    assert "problem_id mismatch across candidates and selections" in message
    assert "candidate_only" in message
    assert "selection_only" in message


def test_selective_examples_can_explicitly_allow_partial_overlap(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections = tmp_path / "selections.jsonl"
    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["a", "b"], "labels": [0, 1]},
            {"problem_id": "candidate_only", "candidates": ["a"], "labels": [1]},
        ],
    )
    _write_jsonl(
        selections,
        [
            {"problem_id": "p1", "chosen_index": 1, "scores": [-1.0, 2.0]},
            {"problem_id": "selection_only", "chosen_index": 0, "scores": [1.0]},
        ],
    )

    examples = load_selective_examples(
        candidates_path=candidates,
        selections_path=selections,
        allow_partial_overlap=True,
    )

    assert [example.problem_id for example in examples] == ["p1"]


def test_nlverifier_selective_cli_writes_outputs(tmp_path):
    scores = tmp_path / "scores.jsonl"
    output_json = tmp_path / "selective.json"
    output_md = tmp_path / "selective.md"
    _write_jsonl(
        scores,
        [
            {
                "problem_id": "p1",
                "candidates": ["a", "b"],
                "labels": [1, 0],
                "scores": [2.0, 0.0],
                "chosen_index": 0,
            },
            {
                "problem_id": "p2",
                "candidates": ["a", "b"],
                "labels": [0, 1],
                "scores": [1.5, 1.0],
                "chosen_index": 0,
            },
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.nlverifier_selective",
            "--scores",
            str(scores),
            "--confidence-key",
            "probability_margin",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["dataset"]["problems"] == 2
    assert "Selective Risk" in output_md.read_text(encoding="utf-8")
