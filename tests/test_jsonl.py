import pytest

from beqcritic.jsonl import (
    JsonlError,
    load_jsonl_map_by_key,
    load_jsonl_map_by_problem_id,
    load_jsonl_objects,
    matching_problem_ids,
    matching_problem_ids_many,
)


def test_load_jsonl_objects_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8")

    assert load_jsonl_objects(path) == [{"id": "a"}, {"id": "b"}]


def test_load_jsonl_objects_accepts_utf8_bom_by_default(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "a"}\n', encoding="utf-8-sig")

    assert load_jsonl_objects(path) == [{"id": "a"}]


def test_load_jsonl_objects_reports_invalid_json_line(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": bad}\n', encoding="utf-8")

    with pytest.raises(JsonlError) as excinfo:
        load_jsonl_objects(path)

    assert f"{path}:3" in str(excinfo.value)
    assert "Invalid JSON" in str(excinfo.value)


def test_load_jsonl_objects_requires_object_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "a"}\n[1, 2, 3]\n', encoding="utf-8")

    with pytest.raises(JsonlError) as excinfo:
        load_jsonl_objects(path)

    assert f"{path}:2" in str(excinfo.value)
    assert "Expected JSON object" in str(excinfo.value)


def test_load_jsonl_map_by_problem_id_reports_physical_line(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"problem_id": "p1"}\n\n{"missing": true}\n', encoding="utf-8")

    with pytest.raises(JsonlError) as excinfo:
        load_jsonl_map_by_problem_id(path)

    assert f"{path}:3" in str(excinfo.value)
    assert "missing problem_id" in str(excinfo.value)


def test_load_jsonl_map_by_key_rejects_missing_key(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "p1"}\n{"missing": true}\n', encoding="utf-8")

    with pytest.raises(JsonlError) as excinfo:
        load_jsonl_map_by_key(path, "id")

    message = str(excinfo.value)
    assert f"{path}:2" in message
    assert "missing id" in message


def test_load_jsonl_map_by_key_can_require_non_empty_string_keys(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": 1}\n', encoding="utf-8")

    with pytest.raises(JsonlError) as excinfo:
        load_jsonl_map_by_key(path, "id", require_nonempty_string_key=True)

    message = str(excinfo.value)
    assert f"{path}:1" in message
    assert "expected id to be a non-empty string" in message


def test_load_jsonl_map_by_problem_id_rejects_non_string_ids(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"problem_id": 1}\n', encoding="utf-8")

    with pytest.raises(JsonlError) as excinfo:
        load_jsonl_map_by_problem_id(path)

    message = str(excinfo.value)
    assert f"{path}:1" in message
    assert "expected problem_id to be a non-empty string" in message


def test_load_jsonl_map_by_problem_id_rejects_empty_ids(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"problem_id": ""}\n', encoding="utf-8")

    with pytest.raises(JsonlError) as excinfo:
        load_jsonl_map_by_problem_id(path)

    message = str(excinfo.value)
    assert f"{path}:1" in message
    assert "expected problem_id to be a non-empty string" in message


def test_load_jsonl_map_by_problem_id_rejects_duplicates(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"problem_id": "p1", "value": 1}\n\n{"problem_id": "p1", "value": 2}\n',
        encoding="utf-8",
    )

    with pytest.raises(JsonlError) as excinfo:
        load_jsonl_map_by_problem_id(path)

    message = str(excinfo.value)
    assert f"{path}:3" in message
    assert "first seen at line 1" in message
    assert "duplicate problem_id 'p1'" in message


def test_matching_problem_ids_rejects_unmatched_ids_by_default():
    with pytest.raises(ValueError) as excinfo:
        matching_problem_ids(
            {"p1": {}, "left_only": {}},
            {"p1": {}, "right_only": {}},
            left_name="left",
            right_name="right",
        )

    message = str(excinfo.value)
    assert "problem_id mismatch across left and right" in message
    assert "left_only" in message
    assert "right_only" in message


def test_matching_problem_ids_can_return_partial_overlap():
    assert matching_problem_ids(
        {"p1": {}, "left_only": {}},
        {"p1": {}, "right_only": {}},
        left_name="left",
        right_name="right",
        allow_partial_overlap=True,
    ) == ["p1"]


def test_matching_problem_ids_many_rejects_unmatched_ids_by_default():
    with pytest.raises(ValueError) as excinfo:
        matching_problem_ids_many(
            {
                "candidates": {"p1": {}, "candidate_only": {}},
                "selection_a": {"p1": {}, "a_only": {}},
                "selection_b": {"p1": {}, "b_only": {}},
            }
        )

    message = str(excinfo.value)
    assert "problem_id mismatch across candidates, selection_a, selection_b" in message
    assert "candidate_only" in message
    assert "a_only" in message
    assert "b_only" in message


def test_matching_problem_ids_many_can_return_partial_overlap():
    assert matching_problem_ids_many(
        {
            "candidates": {"p1": {}, "candidate_only": {}},
            "selection_a": {"p1": {}, "a_only": {}},
            "selection_b": {"p1": {}, "b_only": {}},
        },
        allow_partial_overlap=True,
    ) == ["p1"]
