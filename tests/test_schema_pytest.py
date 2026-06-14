import pytest

from beqcritic.schema import (
    SchemaError,
    load_grouped_candidates_jsonl,
    validate_grouped_candidates,
)


def test_grouped_candidates_accepts_minimal_record_without_labels():
    rec = {"problem_id": "p1", "candidates": ["theorem t : True := by trivial"]}
    out = validate_grouped_candidates(rec, require_labels=False)
    assert out.problem_id == "p1"
    assert out.candidates == ["theorem t : True := by trivial"]
    assert out.labels is None


def test_grouped_candidates_requires_labels_when_requested():
    rec = {"problem_id": "p1", "candidates": ["a"]}
    with pytest.raises(SchemaError):
        validate_grouped_candidates(rec, require_labels=True)


def test_grouped_candidates_validates_label_shape_and_values():
    rec = {"problem_id": "p1", "candidates": ["a", "b"], "labels": [1, 0]}
    out = validate_grouped_candidates(rec, require_labels=True)
    assert out.labels == [1, 0]

    bad_len = {"problem_id": "p1", "candidates": ["a", "b"], "labels": [1]}
    with pytest.raises(SchemaError):
        validate_grouped_candidates(bad_len, require_labels=True)

    bad_val = {"problem_id": "p1", "candidates": ["a"], "labels": [2]}
    with pytest.raises(SchemaError):
        validate_grouped_candidates(bad_val, require_labels=True)


def test_load_grouped_candidates_jsonl_rejects_duplicate_problem_ids(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text(
        '{"problem_id": "p1", "candidates": ["a"]}\n'
        '{"problem_id": "p1", "candidates": ["b"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(SchemaError) as excinfo:
        load_grouped_candidates_jsonl(path)

    message = str(excinfo.value)
    assert f"{path}:2" in message
    assert "duplicate problem_id 'p1'" in message
    assert "first seen at line 1" in message

