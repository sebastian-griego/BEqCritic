from __future__ import annotations

import json
from math import isclose

import pytest

from beqcritic.jsonl import JsonlError
from beqcritic.verifier_select import _load_input_rows, _load_score_temperature, _selection_confidence
from beqcritic import verifier_select


def test_selection_confidence_reports_probability_and_margin():
    conf = _selection_confidence([2.0, 0.0, -1.0], 0, temperature=2.0)

    assert conf["score_temperature"] == 2.0
    assert conf["runner_up_index"] == 1
    assert isclose(conf["chosen_probability"], 0.7310585786, abs_tol=1e-9)
    assert isclose(conf["runner_up_probability"], 0.5, abs_tol=1e-12)
    assert conf["score_margin"] == 2.0
    assert isclose(conf["probability_margin"], 0.2310585786, abs_tol=1e-9)


def test_selection_confidence_honors_minimize_and_eligible_indices():
    conf = _selection_confidence(
        [0.1, 0.2, 5.0],
        1,
        temperature=1.0,
        minimize=True,
        eligible_indices=[1, 2],
    )

    assert conf["eligible_count"] == 2
    assert conf["runner_up_index"] == 2
    assert conf["score_margin"] == 4.8
    assert conf["chosen_probability"] > conf["runner_up_probability"]


def test_selection_confidence_handles_single_candidate():
    conf = _selection_confidence([3.0], 0)

    assert conf["runner_up_index"] is None
    assert conf["probability_margin"] is None
    assert conf["eligible_count"] == 1


def test_load_score_temperature_from_calibration_json(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps({"temperature": {"input": 1.0, "fitted": 2.3794}}),
        encoding="utf-8",
    )

    assert _load_score_temperature(str(path), 1.0) == 2.3794


def test_selection_confidence_rejects_bad_temperature():
    with pytest.raises(ValueError):
        _selection_confidence([1.0, 0.0], 0, temperature=0.0)


def test_load_input_rows_rejects_typecheck_length_mismatch(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text(
        json.dumps(
            {
                "problem_id": "p1",
                "candidates": ["a", "b"],
                "typechecks": [True],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        _load_input_rows(
            path,
            problem_id_key="problem_id",
            candidates_key="candidates",
            typechecks_key="typechecks",
        )
    assert f"{path}:1" in str(excinfo.value)


def test_verifier_select_rejects_malformed_jsonl_before_writing_output(tmp_path):
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "selections.jsonl"
    input_path.write_text(
        '{"problem_id": "p1", "candidates": ["a"]}\n{"problem_id": bad}\n',
        encoding="utf-8",
    )

    with pytest.raises(JsonlError) as excinfo:
        verifier_select.main(
            [
                "--model",
                "unused-model",
                "--dataset",
                "unused-dataset",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
        )

    assert f"{input_path}:2" in str(excinfo.value)
    assert not output_path.exists()
