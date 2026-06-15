from __future__ import annotations

import json

import pytest

from beqcritic import verifier_cluster_select
from beqcritic.jsonl import JsonlError
from beqcritic.verifier_cluster_select import _load_input_rows


def test_load_input_rows_rejects_duplicate_problem_ids(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text(
        json.dumps({"problem_id": "p1", "candidates": ["theorem a : True"]})
        + "\n"
        + json.dumps({"problem_id": "p1", "candidates": ["theorem b : True"]})
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

    message = str(excinfo.value)
    assert f"{path}:2" in message
    assert "duplicate problem_id 'p1'" in message
    assert "first seen at line 1" in message


def test_load_input_rows_rejects_candidate_and_typecheck_shape_errors(tmp_path):
    bad_candidate = tmp_path / "bad_candidate.jsonl"
    bad_candidate.write_text(
        json.dumps({"problem_id": "p1", "candidates": ["a", 3]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"candidates\[1\].*string"):
        _load_input_rows(
            bad_candidate,
            problem_id_key="problem_id",
            candidates_key="candidates",
            typechecks_key="typechecks",
        )

    bad_typechecks = tmp_path / "bad_typechecks.jsonl"
    bad_typechecks.write_text(
        json.dumps(
            {
                "problem_id": "p1",
                "candidates": ["a", "b"],
                "typechecks": [True, "yes"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"typechecks\[1\].*boolean"):
        _load_input_rows(
            bad_typechecks,
            problem_id_key="problem_id",
            candidates_key="candidates",
            typechecks_key="typechecks",
        )


def test_verifier_cluster_select_rejects_malformed_jsonl_before_output(tmp_path):
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "selection.jsonl"
    input_path.write_text(
        '{"problem_id": "p1", "candidates": ["a"]}\n{"problem_id": bad}\n',
        encoding="utf-8",
    )

    with pytest.raises(JsonlError) as excinfo:
        verifier_cluster_select.main(
            [
                "--verifier-model",
                "unused-verifier",
                "--beqcritic-model",
                "unused-critic",
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
