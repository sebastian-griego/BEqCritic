from __future__ import annotations

import json
import subprocess
import sys

from beqcritic.nlverifier_abstain import (
    apply_abstention,
    load_source_rows,
    resolve_threshold,
)
from beqcritic.nlverifier_selective import load_selective_examples


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_apply_abstention_splits_accepted_and_abstained(tmp_path):
    scores = tmp_path / "scores.jsonl"
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
                "labels": [0, 1],
                "scores": [0.1, 0.0],
                "chosen_index": 0,
            },
        ],
    )
    examples = load_selective_examples(scored_path=scores)
    source_rows = load_source_rows(scored_path=scores)

    accepted, abstained, report = apply_abstention(
        examples,
        source_rows,
        confidence_key="chosen_probability",
        threshold=0.7,
    )

    assert [row["problem_id"] for row in accepted] == ["p1"]
    assert [row["problem_id"] for row in abstained] == ["p2"]
    assert accepted[0]["chosen"] == "a"
    assert report["dataset"]["accepted"] == 1
    assert report["dataset"]["accepted_correct"] == 1
    assert report["dataset"]["abstained_correct"] == 0


def test_resolve_threshold_from_target_recommendation(tmp_path):
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        json.dumps(
            {
                "signals": {
                    "chosen_probability": {
                        "target_recommendations": [
                            {
                                "target_accuracy": 0.5,
                                "threshold": 0.61,
                                "meets_target": True,
                                "accepted": 4,
                                "coverage": 0.8,
                                "accuracy": 0.75,
                            }
                        ],
                        "best_lower_bound_prefix": {
                            "threshold": 0.7,
                            "accepted": 3,
                            "coverage": 0.6,
                            "accuracy": 1.0,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_threshold(
        thresholds_json=thresholds,
        confidence_key="chosen_probability",
        target_accuracy=0.5,
        require_certified=True,
    )

    assert resolved["threshold"] == 0.61
    assert resolved["source"] == "target_recommendation"
    assert resolved["meets_target"] is True


def test_nlverifier_abstain_cli_writes_outputs(tmp_path):
    scores = tmp_path / "scores.jsonl"
    output_accepted = tmp_path / "accepted.jsonl"
    output_abstained = tmp_path / "abstained.jsonl"
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
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
                "labels": [0, 1],
                "scores": [0.1, 0.0],
                "chosen_index": 0,
            },
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.nlverifier_abstain",
            "--scores",
            str(scores),
            "--threshold",
            "0.7",
            "--output-accepted",
            str(output_accepted),
            "--output-abstained",
            str(output_abstained),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )

    accepted = [json.loads(line) for line in output_accepted.read_text(encoding="utf-8").splitlines()]
    abstained = [json.loads(line) for line in output_abstained.read_text(encoding="utf-8").splitlines()]
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(accepted) == 1
    assert len(abstained) == 1
    assert payload["dataset"]["accepted"] == 1
    assert "Abstention Policy" in output_md.read_text(encoding="utf-8")
