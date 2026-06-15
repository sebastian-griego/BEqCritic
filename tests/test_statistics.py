from __future__ import annotations

import json
from math import isclose
import subprocess
import sys

import pytest

from beqcritic.compare_selection_methods import compare_files, format_markdown
from beqcritic.statistics import (
    exact_two_sided_sign_test,
    paired_comparison,
    proportion_summary,
    wilson_interval,
)


def test_wilson_interval_and_proportion_summary():
    low, high = wilson_interval(5, 10)

    assert isclose(low, 0.2366, abs_tol=1e-4)
    assert isclose(high, 0.7634, abs_tol=1e-4)

    summary = proportion_summary(5, 10).to_json_dict()
    assert summary["successes"] == 5
    assert summary["total"] == 10
    assert summary["rate"] == 0.5
    assert summary["ci_low"] == low
    assert summary["ci_high"] == high


def test_exact_sign_test_and_paired_comparison():
    assert isclose(exact_two_sided_sign_test(wins=8, losses=2), 0.109375, abs_tol=1e-12)
    assert exact_two_sided_sign_test(wins=0, losses=0) == 1.0

    comparison = paired_comparison(
        [True, True, False, False, False],
        [True, False, True, True, False],
    )
    assert comparison.both_success == 1
    assert comparison.a_only == 1
    assert comparison.b_only == 2
    assert comparison.neither_success == 1
    assert comparison.b_minus_a == 0.2
    assert comparison.discordant == 3


def test_compare_selection_methods_cli_and_markdown(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections_a = tmp_path / "a.jsonl"
    selections_b = tmp_path / "b.jsonl"
    summary_json = tmp_path / "summary.json"
    summary_md = tmp_path / "summary.md"

    candidates.write_text(
        "\n".join(
            [
                json.dumps(
                    {"problem_id": "p1", "candidates": ["a", "b"], "labels": [1, 0]}
                ),
                json.dumps(
                    {"problem_id": "p2", "candidates": ["a", "b"], "labels": [0, 1]}
                ),
                json.dumps(
                    {"problem_id": "p3", "candidates": ["a", "b"], "labels": [0, 1]}
                ),
                json.dumps(
                    {"problem_id": "p4", "candidates": ["a", "b"], "labels": [0, 0]}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    selections_a.write_text(
        "\n".join(
            [
                json.dumps({"problem_id": "p1", "chosen_index": 0}),
                json.dumps({"problem_id": "p2", "chosen_index": 0}),
                json.dumps({"problem_id": "p3", "chosen_index": 0}),
                json.dumps({"problem_id": "p4", "chosen_index": 0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    selections_b.write_text(
        "\n".join(
            [
                json.dumps({"problem_id": "p1", "chosen_index": 1}),
                json.dumps({"problem_id": "p2", "chosen_index": 1}),
                json.dumps({"problem_id": "p3", "chosen_indices": [1, 0]}),
                json.dumps({"problem_id": "p4", "chosen_index": 0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = compare_files(
        candidates_path=candidates,
        selections_a_path=selections_a,
        selections_b_path=selections_b,
        a_name="baseline",
        b_name="critic",
    )
    assert summary["dataset"]["problems"] == 4
    assert summary["dataset"]["has_any_correct"]["successes"] == 3
    assert summary["a"]["selected_correct"]["successes"] == 1
    assert summary["b"]["selected_correct"]["successes"] == 2
    assert summary["paired"]["a_only"] == 1
    assert summary["paired"]["b_only"] == 2
    assert "Exact two-sided sign-test" in format_markdown(summary)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.compare_selection_methods",
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
            str(summary_json),
            "--output-md",
            str(summary_md),
        ],
        check=True,
    )

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["paired"]["b_minus_a"] == 0.25
    assert "Selection comparison: baseline vs critic" in summary_md.read_text(
        encoding="utf-8"
    )


def test_compare_selection_methods_rejects_duplicate_problem_ids(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections_a = tmp_path / "a.jsonl"
    selections_b = tmp_path / "b.jsonl"

    candidates.write_text(
        json.dumps({"problem_id": "p1", "candidates": ["a"], "labels": [1]})
        + "\n"
        + json.dumps({"problem_id": "p1", "candidates": ["b"], "labels": [0]})
        + "\n",
        encoding="utf-8",
    )
    selections_a.write_text(json.dumps({"problem_id": "p1", "chosen_index": 0}) + "\n", encoding="utf-8")
    selections_b.write_text(json.dumps({"problem_id": "p1", "chosen_index": 0}) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.compare_selection_methods",
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
    assert "duplicate problem_id 'p1'" in proc.stderr
    assert f"{candidates}:2" in proc.stderr


def test_compare_selection_methods_rejects_unmatched_problem_ids(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections_a = tmp_path / "a.jsonl"
    selections_b = tmp_path / "b.jsonl"
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

    with pytest.raises(ValueError) as excinfo:
        compare_files(
            candidates_path=candidates,
            selections_a_path=selections_a,
            selections_b_path=selections_b,
            a_name="baseline",
            b_name="critic",
        )

    message = str(excinfo.value)
    assert "problem_id mismatch across candidates" in message
    assert "candidate_only" in message
    assert "a_only" in message
    assert "b_only" in message


def test_compare_selection_methods_can_explicitly_allow_partial_overlap(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections_a = tmp_path / "a.jsonl"
    selections_b = tmp_path / "b.jsonl"
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

    summary = compare_files(
        candidates_path=candidates,
        selections_a_path=selections_a,
        selections_b_path=selections_b,
        allow_partial_overlap=True,
    )

    assert summary["dataset"]["problems"] == 1


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
