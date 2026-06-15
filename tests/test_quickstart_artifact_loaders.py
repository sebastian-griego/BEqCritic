from __future__ import annotations

import json
import subprocess
import sys

import pytest

from beqcritic.jsonl import JsonlError
from scripts import generate_results


def test_evaluate_ab_accepts_chosen_indices_and_writes_summary(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections_a = tmp_path / "a.jsonl"
    selections_b = tmp_path / "b.jsonl"
    output_json = tmp_path / "summary.json"

    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["a", "b"], "labels": [1, 0]},
            {"problem_id": "p2", "candidates": ["a", "b"], "labels": [0, 1]},
        ],
    )
    _write_jsonl(
        selections_a,
        [
            {"problem_id": "p1", "chosen_indices": [0, 1]},
            {"problem_id": "p2", "chosen_indices": [0, 1]},
        ],
    )
    _write_jsonl(
        selections_b,
        [
            {"problem_id": "p1", "chosen_index": 1},
            {"problem_id": "p2", "chosen_index": 1},
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.evaluate_ab",
            "--candidates",
            str(candidates),
            "--selections-a",
            str(selections_a),
            "--selections-b",
            str(selections_b),
            "--a-name",
            "baseline",
            "--b-name",
            "critic",
            "--output-json",
            str(output_json),
        ],
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["a"]["selected_correct_pct"] == 50.0
    assert payload["b"]["selected_correct_pct"] == 50.0
    assert payload["b_minus_a"]["selected_correct_pct"] == 0.0


def test_evaluate_ab_rejects_duplicate_problem_ids_with_line_number(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections_a = tmp_path / "a.jsonl"
    selections_b = tmp_path / "b.jsonl"

    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["a"], "labels": [1]},
            {"problem_id": "p1", "candidates": ["b"], "labels": [0]},
        ],
    )
    _write_jsonl(selections_a, [{"problem_id": "p1", "chosen_index": 0}])
    _write_jsonl(selections_b, [{"problem_id": "p1", "chosen_index": 0}])

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.evaluate_ab",
            "--candidates",
            str(candidates),
            "--selections-a",
            str(selections_a),
            "--selections-b",
            str(selections_b),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert f"{candidates}:2" in proc.stderr
    assert "duplicate problem_id 'p1'" in proc.stderr


def test_evaluate_ab_rejects_unmatched_problem_ids_before_writing_summary(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections_a = tmp_path / "a.jsonl"
    selections_b = tmp_path / "b.jsonl"
    output_json = tmp_path / "summary.json"

    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["a"], "labels": [1]},
            {"problem_id": "candidate_only", "candidates": ["a"], "labels": [1]},
        ],
    )
    _write_jsonl(
        selections_a,
        [
            {"problem_id": "p1", "chosen_index": 0},
            {"problem_id": "a_only", "chosen_index": 0},
        ],
    )
    _write_jsonl(
        selections_b,
        [
            {"problem_id": "p1", "chosen_index": 0},
            {"problem_id": "b_only", "chosen_index": 0},
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.evaluate_ab",
            "--candidates",
            str(candidates),
            "--selections-a",
            str(selections_a),
            "--selections-b",
            str(selections_b),
            "--output-json",
            str(output_json),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "problem_id mismatch across candidates" in proc.stderr
    assert "candidate_only" in proc.stderr
    assert "a_only" in proc.stderr
    assert "b_only" in proc.stderr
    assert not output_json.exists()


def test_generate_results_label_loader_rejects_duplicate_problem_ids(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["a"], "labels": [1]},
            {"problem_id": "p1", "candidates": ["b"], "labels": [0]},
        ],
    )

    with pytest.raises(JsonlError) as excinfo:
        generate_results._load_labels_map(candidates)

    message = str(excinfo.value)
    assert f"{candidates}:2" in message
    assert "duplicate problem_id 'p1'" in message


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
