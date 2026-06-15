from __future__ import annotations

import json
import subprocess
import sys


def test_summarize_selection_rejects_unmatched_problem_ids(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections = tmp_path / "selections.jsonl"
    output_json = tmp_path / "summary.json"
    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["a"], "labels": [1]},
            {"problem_id": "candidate_only", "candidates": ["b"], "labels": [0]},
        ],
    )
    _write_jsonl(
        selections,
        [
            {"problem_id": "p1", "chosen_index": 0},
            {"problem_id": "selection_only", "chosen_index": 0},
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_selection.py",
            "--candidates",
            str(candidates),
            "--selections",
            str(selections),
            "--output-json",
            str(output_json),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "problem_id mismatch across candidates and selections" in proc.stderr
    assert "candidate_only" in proc.stderr
    assert "selection_only" in proc.stderr
    assert not output_json.exists()


def test_summarize_selection_can_explicitly_allow_partial_overlap(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections = tmp_path / "selections.jsonl"
    output_json = tmp_path / "summary.json"
    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["a"], "labels": [1]},
            {"problem_id": "candidate_only", "candidates": ["b"], "labels": [0]},
        ],
    )
    _write_jsonl(
        selections,
        [
            {"problem_id": "p1", "chosen_index": 0},
            {"problem_id": "selection_only", "chosen_index": 0},
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_selection.py",
            "--candidates",
            str(candidates),
            "--selections",
            str(selections),
            "--output-json",
            str(output_json),
            "--allow-partial-overlap",
        ],
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["problems"] == 1
    assert payload["selected_correct_pct"] == 100.0


def test_summarize_selection_rejects_duplicate_problem_ids(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections = tmp_path / "selections.jsonl"
    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["a"], "labels": [1]},
            {"problem_id": "p1", "candidates": ["b"], "labels": [0]},
        ],
    )
    _write_jsonl(selections, [{"problem_id": "p1", "chosen_index": 0}])

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_selection.py",
            "--candidates",
            str(candidates),
            "--selections",
            str(selections),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert f"{candidates}:2" in proc.stderr
    assert "duplicate problem_id 'p1'" in proc.stderr


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
