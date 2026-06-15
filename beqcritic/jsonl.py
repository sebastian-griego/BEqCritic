from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping


class JsonlError(ValueError):
    """Raised when a JSONL artifact has an invalid row."""


def load_jsonl_objects(path: str | Path, *, encoding: str = "utf-8-sig") -> list[dict[str, Any]]:
    return [row for _line_no, row in iter_jsonl_objects(path, encoding=encoding)]


def load_jsonl_map_by_key(
    path: str | Path,
    key: str,
    *,
    encoding: str = "utf-8-sig",
    require_nonempty_string_key: bool = False,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    first_lines: dict[str, int] = {}
    for line_no, record in iter_jsonl_objects(path, encoding=encoding):
        raw_key = record.get(key)
        if raw_key is None:
            raise JsonlError(f"missing {key} at {Path(path)}:{line_no}")
        if require_nonempty_string_key and (
            not isinstance(raw_key, str) or not raw_key.strip()
        ):
            raise JsonlError(
                f"expected {key} to be a non-empty string at {Path(path)}:{line_no}, "
                f"got {type(raw_key).__name__}"
            )
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
    encoding: str = "utf-8-sig",
) -> dict[str, dict[str, Any]]:
    return load_jsonl_map_by_key(
        path,
        "problem_id",
        encoding=encoding,
        require_nonempty_string_key=True,
    )


def matching_problem_ids(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_name: str,
    right_name: str,
    allow_partial_overlap: bool = False,
) -> list[str]:
    left_ids = set(left)
    right_ids = set(right)
    common = sorted(left_ids & right_ids)
    if not common:
        raise ValueError(f"no overlapping problem_ids across {left_name} and {right_name}")
    if allow_partial_overlap:
        return common

    left_only = sorted(left_ids - right_ids)
    right_only = sorted(right_ids - left_ids)
    if left_only or right_only:
        parts = []
        if left_only:
            parts.append(
                f"{len(left_only)} {left_name} problem_ids missing from {right_name}: "
                f"{_preview_ids(left_only)}"
            )
        if right_only:
            parts.append(
                f"{len(right_only)} {right_name} problem_ids missing from {left_name}: "
                f"{_preview_ids(right_only)}"
            )
        raise ValueError(
            f"problem_id mismatch across {left_name} and {right_name}; "
            + "; ".join(parts)
        )
    return common


def matching_problem_ids_many(
    named_maps: Mapping[str, Mapping[str, Any]],
    *,
    allow_partial_overlap: bool = False,
) -> list[str]:
    items = [(str(name), mapping) for name, mapping in named_maps.items()]
    if not items:
        raise ValueError("at least one problem_id map is required")

    id_sets = {name: set(mapping) for name, mapping in items}
    common = sorted(set.intersection(*(ids for _name, ids in id_sets.items())))
    names = ", ".join(name for name, _mapping in items)
    if not common:
        raise ValueError(f"no overlapping problem_ids across {names}")
    if allow_partial_overlap:
        return common

    union_ids = set.union(*(ids for _name, ids in id_sets.items()))
    parts = []
    for name, ids in id_sets.items():
        missing = sorted(union_ids - ids)
        if missing:
            parts.append(
                f"{len(missing)} problem_ids missing from {name}: "
                f"{_preview_ids(missing)}"
            )
    if parts:
        raise ValueError(f"problem_id mismatch across {names}; " + "; ".join(parts))
    return common


def iter_jsonl_objects(
    path: str | Path,
    *,
    encoding: str = "utf-8-sig",
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


def _preview_ids(ids: list[str], limit: int = 5) -> str:
    shown = ", ".join(repr(item) for item in ids[:limit])
    if len(ids) > limit:
        shown += f", ... ({len(ids)} total)"
    return shown


__all__ = [
    "JsonlError",
    "iter_jsonl_objects",
    "load_jsonl_map_by_key",
    "load_jsonl_map_by_problem_id",
    "load_jsonl_objects",
    "matching_problem_ids",
    "matching_problem_ids_many",
]
