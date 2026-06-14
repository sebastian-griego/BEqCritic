from __future__ import annotations

import json
from math import isclose

import pytest

from beqcritic.verifier_select import _load_score_temperature, _selection_confidence


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
