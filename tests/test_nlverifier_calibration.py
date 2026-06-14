from __future__ import annotations

import json
import subprocess
import sys

from beqcritic.nlverifier_calibration import (
    analyze_calibration,
    fit_temperature_for_nll,
    load_scored_examples,
)


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_calibration_metrics_fit_temperature_and_bins(tmp_path):
    scores = tmp_path / "scores.jsonl"
    _write_jsonl(
        scores,
        [
            {
                "problem_id": "p1",
                "candidates": ["a", "b", "c"],
                "labels": [1, 0, 0],
                "scores": [1.0, 0.6, -1.0],
            },
            {
                "problem_id": "p2",
                "candidates": ["a", "b"],
                "labels": [0, 1],
                "scores": [0.2, 1.2],
            },
        ],
    )

    examples = load_scored_examples(scored_path=scores)
    summary = analyze_calibration(examples, bins=5, fit_temperature=True)

    assert summary["dataset"]["problems"] == 2
    assert summary["dataset"]["candidates"] == 5
    assert summary["dataset"]["positive"] == 2
    assert summary["temperature"]["fitted"] > 0
    assert summary["after"]["nll"] <= summary["before"]["nll"]
    assert len(summary["after"]["reliability_bins"]) == 5
    assert summary["thresholds"]["best_f1"]["f1"] >= summary["thresholds"]["at_0_5"]["f1"]


def test_load_scored_examples_from_candidate_selection_join(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections = tmp_path / "selections.jsonl"
    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["a", "b"], "labels": [0, 1]},
        ],
    )
    _write_jsonl(
        selections,
        [
            {"problem_id": "p1", "chosen_index": 1, "scores": [-1.0, 2.0]},
        ],
    )

    examples = load_scored_examples(candidates_path=candidates, selections_path=selections)

    assert [(e.problem_id, e.candidate_index, e.label, e.score) for e in examples] == [
        ("p1", 0, 0, -1.0),
        ("p1", 1, 1, 2.0),
    ]


def test_fit_temperature_reduces_loss_for_overconfident_scores():
    scores = [8.0, 7.0, -8.0, -7.0, 8.0, -8.0]
    labels = [1, 0, 0, 1, 1, 0]

    fitted = fit_temperature_for_nll(scores, labels)

    assert fitted > 1.0


def test_nlverifier_calibration_cli_writes_outputs(tmp_path):
    scores = tmp_path / "scores.jsonl"
    output_json = tmp_path / "calibration.json"
    output_md = tmp_path / "calibration.md"
    _write_jsonl(
        scores,
        [
            {
                "problem_id": "p1",
                "candidates": ["a", "b"],
                "labels": [1, 0],
                "scores": [1.5, -1.5],
            },
            {
                "problem_id": "p2",
                "candidates": ["a", "b"],
                "labels": [0, 1],
                "scores": [-1.2, 1.0],
            },
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.nlverifier_calibration",
            "--scores",
            str(scores),
            "--fit-temperature",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["dataset"]["candidates"] == 4
    assert "NLVerifier Calibration" in output_md.read_text(encoding="utf-8")
