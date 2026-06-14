from __future__ import annotations

import json
import subprocess
import sys

from beqcritic.nlverifier_selective import load_selective_examples
from beqcritic.nlverifier_thresholds import analyze_thresholds, format_markdown


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_threshold_recommendations_use_wilson_lower_bound(tmp_path):
    scores = tmp_path / "scores.jsonl"
    rows = []
    for idx in range(10):
        rows.append(
            {
                "problem_id": f"p{idx}",
                "candidates": ["a", "b"],
                "labels": [1 if idx < 9 else 0, 0 if idx < 9 else 1],
                "scores": [10.0 - idx, -1.0],
                "chosen_index": 0,
            }
        )
    _write_jsonl(scores, rows)

    examples = load_selective_examples(scored_path=scores)
    summary = analyze_thresholds(
        examples,
        target_accuracies=[0.55, 0.8],
        min_accepted=5,
    )

    signal = summary["signals"]["chosen_probability"]
    target_55, target_80 = signal["target_recommendations"]
    assert target_55["meets_target"] is True
    assert target_55["accepted"] == 10
    assert target_80["meets_target"] is False
    assert target_80["accepted"] >= 5
    assert target_80["accuracy_ci95"]["low"] < 0.8
    assert "Threshold Recommendations" in format_markdown(summary)


def test_nlverifier_thresholds_cli_writes_outputs(tmp_path):
    scores = tmp_path / "scores.jsonl"
    output_json = tmp_path / "thresholds.json"
    output_md = tmp_path / "thresholds.md"
    _write_jsonl(
        scores,
        [
            {
                "problem_id": "p1",
                "candidates": ["a", "b"],
                "labels": [1, 0],
                "scores": [4.0, 0.0],
                "chosen_index": 0,
            },
            {
                "problem_id": "p2",
                "candidates": ["a", "b"],
                "labels": [1, 0],
                "scores": [3.0, 0.0],
                "chosen_index": 0,
            },
            {
                "problem_id": "p3",
                "candidates": ["a", "b"],
                "labels": [0, 1],
                "scores": [2.0, 1.0],
                "chosen_index": 0,
            },
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.nlverifier_thresholds",
            "--scores",
            str(scores),
            "--target-accuracies",
            "0.4",
            "--min-accepted",
            "2",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["dataset"]["problems"] == 3
    assert payload["signals"]["chosen_probability"]["target_recommendations"][0]["accepted"] >= 2
    assert "Recommendations" in output_md.read_text(encoding="utf-8")
