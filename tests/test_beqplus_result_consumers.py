from __future__ import annotations

import json
import subprocess
import sys

import pytest

from beqcritic.jsonl import JsonlError
from scripts import summarize_ab


def test_summarize_ab_rejects_duplicate_problem_ids(tmp_path):
    path = tmp_path / "beqplus_results.jsonl"
    _write_jsonl(
        path,
        [
            {"problem_id": "p1", "a_ok": True, "b_ok": False},
            {"problem_id": "p1", "a_ok": False, "b_ok": True},
        ],
    )

    with pytest.raises(JsonlError) as excinfo:
        summarize_ab._load_rows(str(path))

    message = str(excinfo.value)
    assert f"{path}:2" in message
    assert "duplicate problem_id 'p1'" in message


def test_summarize_ab_cli_writes_paired_summary(tmp_path):
    path = tmp_path / "beqplus_results.jsonl"
    output_json = tmp_path / "summary.json"
    _write_jsonl(
        path,
        [
            {"problem_id": "p1", "a_ok": True, "b_ok": True, "a_name": "selfbleu", "b_name": "critic"},
            {"problem_id": "p2", "a_ok": True, "b_ok": False, "a_name": "selfbleu", "b_name": "critic"},
            {"problem_id": "p3", "a_ok": False, "b_ok": True, "a_name": "selfbleu", "b_name": "critic"},
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_ab.py",
            "--input",
            str(path),
            "--output-json",
            str(output_json),
        ],
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["problems"] == 3
    assert payload["a"]["hits"] == 2
    assert payload["b"]["hits"] == 2
    assert payload["contingency"] == {"both": 1, "a_only": 1, "b_only": 1, "neither": 0}


def test_beqplus_vs_labels_accepts_chosen_indices_and_writes_joined_rows(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections_a = tmp_path / "a.jsonl"
    selections_b = tmp_path / "b.jsonl"
    results = tmp_path / "beqplus_results.jsonl"
    output_jsonl = tmp_path / "joined.jsonl"

    _write_jsonl(
        candidates,
        [
            {
                "problem_id": "p1",
                "candidates": [
                    "theorem p1 : True := by trivial",
                    "theorem p1 : False := by sorry",
                ],
                "labels": [1, 0],
            }
        ],
    )
    _write_jsonl(selections_a, [{"problem_id": "p1", "chosen_indices": [0, 1]}])
    _write_jsonl(selections_b, [{"problem_id": "p1", "chosen_index": 1}])
    _write_jsonl(results, [{"problem_id": "p1", "a_ok": True, "b_ok": False}])

    subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.analyze_beqplus_vs_labels",
            "--candidates",
            str(candidates),
            "--selections-a",
            str(selections_a),
            "--selections-b",
            str(selections_b),
            "--beqplus-results",
            str(results),
            "--name-a",
            "selfbleu",
            "--name-b",
            "critic",
            "--output-jsonl",
            str(output_jsonl),
            "--max-examples",
            "0",
        ],
        check=True,
    )

    rows = [
        json.loads(line)
        for line in output_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["problem_id"] == "p1"
    assert rows[0]["a"]["chosen_index"] == 0
    assert rows[0]["a"]["label_ok"] is True
    assert rows[0]["b"]["chosen_index"] == 1
    assert rows[0]["b"]["label_ok"] is False


def test_beqplus_vs_labels_rejects_duplicate_results_with_line_number(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections_a = tmp_path / "a.jsonl"
    selections_b = tmp_path / "b.jsonl"
    results = tmp_path / "beqplus_results.jsonl"

    _write_jsonl(
        candidates,
        [{"problem_id": "p1", "candidates": ["theorem p1 : True := by trivial"], "labels": [1]}],
    )
    _write_jsonl(selections_a, [{"problem_id": "p1", "chosen_index": 0}])
    _write_jsonl(selections_b, [{"problem_id": "p1", "chosen_index": 0}])
    _write_jsonl(
        results,
        [
            {"problem_id": "p1", "a_ok": True, "b_ok": True},
            {"problem_id": "p1", "a_ok": False, "b_ok": False},
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.analyze_beqplus_vs_labels",
            "--candidates",
            str(candidates),
            "--selections-a",
            str(selections_a),
            "--selections-b",
            str(selections_b),
            "--beqplus-results",
            str(results),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert f"{results}:2" in proc.stderr
    assert "duplicate problem_id 'p1'" in proc.stderr


def test_beqplus_vs_labels_rejects_unmatched_problem_ids(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections_a = tmp_path / "a.jsonl"
    selections_b = tmp_path / "b.jsonl"
    results = tmp_path / "beqplus_results.jsonl"
    output_jsonl = tmp_path / "joined.jsonl"

    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["theorem p1 : True := by trivial"], "labels": [1]},
            {"problem_id": "candidate_only", "candidates": ["theorem c : True := by trivial"], "labels": [1]},
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
    _write_jsonl(
        results,
        [
            {"problem_id": "p1", "a_ok": True, "b_ok": True},
            {"problem_id": "result_only", "a_ok": False, "b_ok": False},
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.analyze_beqplus_vs_labels",
            "--candidates",
            str(candidates),
            "--selections-a",
            str(selections_a),
            "--selections-b",
            str(selections_b),
            "--beqplus-results",
            str(results),
            "--output-jsonl",
            str(output_jsonl),
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
    assert "result_only" in proc.stderr
    assert not output_jsonl.exists()


def test_beqplus_vs_labels_can_explicitly_allow_partial_overlap(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections_a = tmp_path / "a.jsonl"
    selections_b = tmp_path / "b.jsonl"
    results = tmp_path / "beqplus_results.jsonl"
    output_jsonl = tmp_path / "joined.jsonl"

    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["theorem p1 : True := by trivial"], "labels": [1]},
            {"problem_id": "candidate_only", "candidates": ["theorem c : True := by trivial"], "labels": [1]},
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
    _write_jsonl(
        results,
        [
            {"problem_id": "p1", "a_ok": True, "b_ok": True},
            {"problem_id": "result_only", "a_ok": False, "b_ok": False},
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.analyze_beqplus_vs_labels",
            "--candidates",
            str(candidates),
            "--selections-a",
            str(selections_a),
            "--selections-b",
            str(selections_b),
            "--beqplus-results",
            str(results),
            "--output-jsonl",
            str(output_jsonl),
            "--allow-partial-overlap",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    rows = [
        json.loads(line)
        for line in output_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["problem_id"] for row in rows] == ["p1"]


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
