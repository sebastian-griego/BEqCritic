import pytest

from beqcritic.jsonl import (
    JsonlError,
    load_jsonl_map_by_key,
    load_jsonl_map_by_problem_id,
    load_jsonl_objects,
)


def test_load_jsonl_objects_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8")

    assert load_jsonl_objects(path) == [{"id": "a"}, {"id": "b"}]


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
