from __future__ import annotations

import json
import subprocess
import sys

from beqcritic.nlverifier_confidence_audit import (
    analyze_confidence_signals,
    format_markdown,
)
from beqcritic.nlverifier_selective import load_selective_examples


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_confidence_audit_compares_signal_quality(tmp_path):
    scores = tmp_path / "scores.jsonl"
    _write_jsonl(
        scores,
        [
            {
                "problem_id": "correct_high_margin",
                "candidates": ["a", "b"],
                "labels": [1, 0],
                "scores": [4.0, 0.0],
                "chosen_index": 0,
            },
            {
                "problem_id": "wrong_high_probability_low_margin",
                "candidates": ["a", "b"],
                "labels": [0, 1],
                "scores": [5.0, 4.9],
                "chosen_index": 0,
            },
            {
                "problem_id": "correct_medium_probability_high_margin",
                "candidates": ["a", "b"],
                "labels": [1, 0],
                "scores": [2.0, -2.0],
                "chosen_index": 0,
            },
        ],
    )

    examples = load_selective_examples(scored_path=scores)
    summary = analyze_confidence_signals(
        examples,
        confidence_keys=["chosen_probability", "probability_margin"],
        coverages=[1 / 3, 2 / 3, 1.0],
    )

    assert summary["dataset"]["problems"] == 3
    assert summary["coverage_comparison"][1]["best_key"] == "probability_margin"
    assert summary["coverage_comparison"][1]["by_key"]["chosen_probability"]["accuracy"] == 0.5
    assert summary["coverage_comparison"][1]["by_key"]["probability_margin"]["accuracy"] == 1.0
    assert summary["best_by_oracle_normalized_accuracy_area"] == "probability_margin"
    assert (
        summary["signals"]["probability_margin"]["ranking_metrics"]["average_precision"]
        > summary["signals"]["chosen_probability"]["ranking_metrics"]["average_precision"]
    )
    assert (
        summary["signals"]["probability_margin"]["ranking_metrics"][
            "area_under_accuracy_coverage"
        ]
        > summary["signals"]["chosen_probability"]["ranking_metrics"][
            "area_under_accuracy_coverage"
        ]
    )
    markdown = format_markdown(summary)
    assert "Confidence Signal Audit" in markdown
    assert "oracle-normalized area" in markdown


def test_nlverifier_confidence_audit_cli_writes_outputs(tmp_path):
    scores = tmp_path / "scores.jsonl"
    output_json = tmp_path / "confidence.json"
    output_md = tmp_path / "confidence.md"
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
            "beqcritic.nlverifier_confidence_audit",
            "--scores",
            str(scores),
            "--confidence-keys",
            "chosen_probability,probability_margin",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["dataset"]["problems"] == 2
    assert payload["confidence_keys"] == ["chosen_probability", "probability_margin"]
    assert "ranking_metrics" in payload["signals"]["chosen_probability"]
    text = output_md.read_text(encoding="utf-8")
    assert "Coverage Comparison" in text
    assert "average precision" in text
