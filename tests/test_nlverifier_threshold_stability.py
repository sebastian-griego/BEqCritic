from __future__ import annotations

import json
import subprocess
import sys

from beqcritic.nlverifier_selective import load_selective_examples
from beqcritic.nlverifier_threshold_stability import (
    analyze_threshold_stability,
    format_markdown,
)


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _rows():
    rows = []
    correct_flags = [True, True, True, False, True, False]
    for idx, selected_correct in enumerate(correct_flags):
        rows.append(
            {
                "problem_id": f"p{idx}",
                "candidates": ["chosen", "other"],
                "labels": [1, 0] if selected_correct else [0, 1],
                "scores": [6.0 - idx, 0.0],
                "chosen_index": 0,
            }
        )
    return rows


def test_threshold_stability_reports_leave_one_out_changes(tmp_path):
    scores = tmp_path / "scores.jsonl"
    _write_jsonl(scores, _rows())

    examples = load_selective_examples(scored_path=scores)
    summary = analyze_threshold_stability(
        examples,
        target_accuracy=0.32,
        min_accepted=2,
        top_cases=3,
    )
    text = format_markdown(summary)

    assert summary["dataset"]["problems"] == 6
    assert summary["leave_one_out"]["resamples"] == 6
    assert summary["leave_one_out"]["unique_threshold_count"] > 1
    assert summary["leave_one_out"]["threshold_changed"] > 0
    assert len(summary["sensitive_omissions"]) == 3
    assert "Threshold Stability" in text
    assert "Most Sensitive Omissions" in text


def test_nlverifier_threshold_stability_cli_writes_outputs(tmp_path):
    scores = tmp_path / "scores.jsonl"
    output_json = tmp_path / "stability.json"
    output_md = tmp_path / "stability.md"
    _write_jsonl(scores, _rows())

    subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.nlverifier_threshold_stability",
            "--scores",
            str(scores),
            "--target-accuracy",
            "0.32",
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
    assert payload["policy"]["confidence_key"] == "chosen_probability"
    assert payload["leave_one_out"]["resamples"] == 6
    assert payload["leave_one_out"]["threshold_changed"] > 0
    assert "Leave-One-Out Summary" in output_md.read_text(encoding="utf-8")
