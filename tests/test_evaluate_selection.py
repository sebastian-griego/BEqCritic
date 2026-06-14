from __future__ import annotations

import json
import subprocess
import sys

from beqcritic.evaluate_selection import evaluate_selection_records, format_summary


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_evaluate_selection_keeps_full_coverage_behavior():
    candidates = {
        "p1": {"candidates": ["a", "b"], "labels": [1, 0]},
        "p2": {"candidates": ["a", "b"], "labels": [0, 1]},
    }
    selections = {
        "p1": {"problem_id": "p1", "chosen_index": 0},
        "p2": {"problem_id": "p2", "chosen_index": 0},
    }

    summary = evaluate_selection_records(candidates, selections)
    text = format_summary(summary)

    assert summary["abstention_mode"] is False
    assert summary["problems"] == 2
    assert summary["accepted"] == 2
    assert summary["selected_correct"]["successes"] == 1
    assert "Selected correct: 1 (50.0%)" in text


def test_missing_selections_can_be_counted_as_abstentions():
    candidates = {
        "p1": {"candidates": ["a", "b"], "labels": [1, 0]},
        "p2": {"candidates": ["a", "b"], "labels": [0, 1]},
        "p3": {"candidates": ["a", "b"], "labels": [0, 0]},
    }
    selections = {"p1": {"problem_id": "p1", "chosen_index": 0}}

    summary = evaluate_selection_records(
        candidates,
        selections,
        treat_missing_as_abstain=True,
    )
    text = format_summary(summary)

    assert summary["abstention_mode"] is True
    assert summary["problems"] == 3
    assert summary["accepted"] == 1
    assert summary["abstained"] == 2
    assert summary["missing_as_abstained"] == 2
    assert summary["coverage"]["successes"] == 1
    assert summary["accepted_selected_correct"]["successes"] == 1
    assert summary["abstained_has_any_correct"]["successes"] == 1
    assert "Accepted selected correct: 1 (100.0%)" in text
    assert "Missing selections counted as abstentions: 2" in text


def test_explicit_abstention_rows_do_not_require_chosen_index():
    candidates = {
        "p1": {"candidates": ["a", "b"], "labels": [1, 0]},
        "p2": {"candidates": ["a", "b"], "labels": [0, 1]},
        "p3": {"candidates": ["a", "b"], "labels": [1, 0]},
    }
    selections = {
        "p1": {"problem_id": "p1", "chosen_index": 0, "accepted": True},
        "p2": {"problem_id": "p2", "abstained": True},
        "p3": {"problem_id": "p3", "accepted": False, "chosen_index": 0},
    }

    summary = evaluate_selection_records(candidates, selections)
    text = format_summary(summary)

    assert summary["abstention_mode"] is True
    assert summary["accepted"] == 1
    assert summary["abstained"] == 2
    assert summary["explicit_abstained"] == 2
    assert summary["abstained_with_choice"] == 1
    assert summary["abstained_selected_correct"]["successes"] == 1
    assert "Abstained selected correct (diagnostic): 1 (100.0%)" in text


def test_evaluate_selection_cli_merges_abstentions_and_writes_summary(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections = tmp_path / "accepted.jsonl"
    abstentions = tmp_path / "abstained.jsonl"
    summary_json = tmp_path / "summary.json"
    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["a", "b"], "labels": [1, 0]},
            {"problem_id": "p2", "candidates": ["a", "b"], "labels": [0, 1]},
        ],
    )
    _write_jsonl(
        selections,
        [{"problem_id": "p1", "chosen_index": 0, "accepted": True}],
    )
    _write_jsonl(
        abstentions,
        [{"problem_id": "p2", "abstained": True}],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.evaluate_selection",
            "--candidates",
            str(candidates),
            "--selections",
            str(selections),
            "--abstentions",
            str(abstentions),
            "--summary-json",
            str(summary_json),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["accepted"] == 1
    assert payload["abstained"] == 1
    assert payload["coverage"]["successes"] == 1
    assert payload["accepted_selected_correct"]["successes"] == 1
    assert "Accepted: 1 (50.0%)" in proc.stdout
