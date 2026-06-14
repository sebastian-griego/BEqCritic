from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class JsonlError(ValueError):
    """Raised when a JSONL artifact has an invalid row."""


def load_jsonl_objects(path: str | Path, *, encoding: str = "utf-8") -> list[dict[str, Any]]:
    return [row for _line_no, row in iter_jsonl_objects(path, encoding=encoding)]


def load_jsonl_map_by_key(
    path: str | Path,
    key: str,
    *,
    encoding: str = "utf-8",
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    first_lines: dict[str, int] = {}
    for line_no, record in iter_jsonl_objects(path, encoding=encoding):
        raw_key = record.get(key)
        if raw_key is None:
            raise JsonlError(f"missing {key} at {Path(path)}:{line_no}")
        map_key = str(raw_key)
        if map_key in records:
            raise JsonlError(
                f"duplicate {key} {map_key!r} at {Path(path)}:{line_no}; "
                f"first seen at line {first_lines[map_key]}"
            )
        records[map_key] = record
        first_lines[map_key] = line_no
    return records


def load_jsonl_map_by_problem_id(
    path: str | Path,
    *,
    encoding: str = "utf-8",
) -> dict[str, dict[str, Any]]:
    return load_jsonl_map_by_key(path, "problem_id", encoding=encoding)


def iter_jsonl_objects(
    path: str | Path,
    *,
    encoding: str = "utf-8",
) -> Iterator[tuple[int, dict[str, Any]]]:
    path = Path(path)
    with path.open("r", encoding=encoding) as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise JsonlError(f"Invalid JSON at {path}:{line_no}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise JsonlError(
                    f"Expected JSON object at {path}:{line_no}, got {type(row).__name__}"
                )
            yield line_no, row


__all__ = [
    "JsonlError",
    "iter_jsonl_objects",
    "load_jsonl_map_by_key",
    "load_jsonl_map_by_problem_id",
    "load_jsonl_objects",
]
