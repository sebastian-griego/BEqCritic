from __future__ import annotations

import json
import subprocess
import sys

from beqcritic.selection_leaderboard import analyze_leaderboard, format_markdown


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_selection_leaderboard_ranks_methods_and_pairs(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    weak = tmp_path / "weak.jsonl"
    strong = tmp_path / "strong.jsonl"
    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["short", "long correct"], "labels": [0, 1]},
            {"problem_id": "p2", "candidates": ["right", "wrong"], "labels": [1, 0]},
            {"problem_id": "p3", "candidates": ["wrong", "right"], "labels": [0, 1]},
            {"problem_id": "p4", "candidates": ["none", "also none"], "labels": [0, 0]},
        ],
    )
    _write_jsonl(
        weak,
        [
            {"problem_id": "p1", "chosen_index": 0},
            {"problem_id": "p2", "chosen_index": 0},
            {"problem_id": "p3", "chosen_index": 0},
            {"problem_id": "p4", "chosen_index": 0},
        ],
    )
    _write_jsonl(
        strong,
        [
            {"problem_id": "p1", "chosen_index": 1},
            {"problem_id": "p2", "chosen_index": 0},
            {"problem_id": "p3", "chosen_index": 1},
            {"problem_id": "p4", "chosen_index": 0},
        ],
    )

    summary = analyze_leaderboard(
        candidates_path=candidates,
        selections={"weak": weak, "strong": strong},
        include_baselines=["first"],
        max_cases=3,
    )
    markdown = format_markdown(summary)

    assert summary["best_method"] == "strong"
    assert summary["methods"]["strong"]["selected_correct"]["successes"] == 3
    assert summary["methods"]["weak"]["missed_available_correct"] == 2
    assert summary["pairwise"]["weak"]["strong"]["b_only"] == 2
    assert len(summary["best_method_cases"]["weak"]) == 2
    assert "Selection Leaderboard" in markdown
    assert "Pairwise Lift Matrix" in markdown


def test_selection_leaderboard_cli_writes_outputs(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    output_json = tmp_path / "leaderboard.json"
    output_md = tmp_path / "leaderboard.md"
    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["a", "b"], "labels": [1, 0]},
            {"problem_id": "p2", "candidates": ["a", "b"], "labels": [0, 1]},
        ],
    )
    _write_jsonl(
        a,
        [
            {"problem_id": "p1", "chosen_index": 0},
            {"problem_id": "p2", "chosen_index": 0},
        ],
    )
    _write_jsonl(
        b,
        [
            {"problem_id": "p1", "chosen_index": 0},
            {"problem_id": "p2", "chosen_index": 1},
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.selection_leaderboard",
            "--candidates",
            str(candidates),
            "--selection",
            f"a={a}",
            "--selection",
            f"b={b}",
            "--include-baseline",
            "shortest",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["best_method"] == "b"
    assert payload["dataset"]["problems"] == 2
    assert "Leaderboard" in output_md.read_text(encoding="utf-8")
