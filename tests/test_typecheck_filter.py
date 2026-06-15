import json
import sys

import pytest

from beqcritic.jsonl import JsonlError
from beqcritic.paper_pipeline import typecheck_filter


def test_typecheck_filter_rejects_malformed_jsonl_before_outputs(tmp_path):
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "out" / "filtered.jsonl"
    errors_path = tmp_path / "out" / "errors.jsonl"
    input_path.write_text(
        json.dumps({"problem_id": "p1", "candidates": []}) + "\n{bad json}\n",
        encoding="utf-8",
    )

    with pytest.raises(JsonlError) as excinfo:
        typecheck_filter.main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--errors-jsonl",
                str(errors_path),
                "--lean-cmd",
                sys.executable,
            ]
        )

    assert f"{input_path}:2" in str(excinfo.value)
    assert not output_path.exists()
    assert not errors_path.exists()


def test_typecheck_filter_writes_lf_outputs_and_creates_parent_dirs(tmp_path):
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "nested" / "filtered.jsonl"
    errors_path = tmp_path / "nested" / "errors.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "problem_id": "p1",
                "candidates": ["# passes under the Python test shim", "raise SystemExit(1)"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    typecheck_filter.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--errors-jsonl",
            str(errors_path),
            "--lean-cmd",
            sys.executable,
            "--jobs",
            "1",
        ]
    )

    output_bytes = output_path.read_bytes()
    errors_bytes = errors_path.read_bytes()
    assert output_bytes.endswith(b"\n")
    assert b"\r\n" not in output_bytes
    assert errors_bytes.endswith(b"\n")
    assert b"\r\n" not in errors_bytes

    output_rows = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    error_rows = [
        json.loads(line) for line in errors_path.read_text(encoding="utf-8").splitlines()
    ]
    assert output_rows[0]["candidates"] == ["# passes under the Python test shim"]
    assert error_rows[0]["candidate_index"] == 1
