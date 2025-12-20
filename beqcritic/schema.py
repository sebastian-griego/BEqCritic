from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class GroupedCandidates:
    problem_id: str
    candidates: list[str]
    labels: list[int] | None = None


def validate_grouped_candidates(obj: dict[str, Any], *, require_labels: bool = False) -> GroupedCandidates:
    """
    Validate a single grouped-candidates JSON object.

    Expected minimal schema:
      {
        "problem_id": str,
        "candidates": [str, ...],
        "labels": [0/1, ...]   # optional unless require_labels=True
      }
    """
    if not isinstance(obj, dict):
        raise SchemaError(f"Expected object, got {type(obj).__name__}")

    if "problem_id" not in obj:
        raise SchemaError("Missing required key: problem_id")
    problem_id = obj["problem_id"]
    if problem_id is None:
        raise SchemaError("problem_id must be a string, got null")
    problem_id = str(problem_id)
    if not problem_id:
        raise SchemaError("problem_id must be non-empty")

    candidates = obj.get("candidates")
    if not isinstance(candidates, list):
        raise SchemaError("candidates must be a list of strings")
    cand_out: list[str] = []
    for i, c in enumerate(candidates):
        if not isinstance(c, str):
            raise SchemaError(f"candidates[{i}] must be a string, got {type(c).__name__}")
        cand_out.append(c)

    labels_raw = obj.get("labels", None)
    if labels_raw is None:
        if require_labels:
            raise SchemaError("Missing required key: labels")
        return GroupedCandidates(problem_id=problem_id, candidates=cand_out, labels=None)

    if not isinstance(labels_raw, list):
        raise SchemaError("labels must be a list of 0/1 values")
    if len(labels_raw) != len(cand_out):
        raise SchemaError(f"labels length {len(labels_raw)} != candidates length {len(cand_out)}")
    labels_out: list[int] = []
    for i, v in enumerate(labels_raw):
        if isinstance(v, bool):
            labels_out.append(1 if v else 0)
            continue
        if isinstance(v, int):
            if v not in (0, 1):
                raise SchemaError(f"labels[{i}] must be 0 or 1, got {v!r}")
            labels_out.append(int(v))
            continue
        raise SchemaError(f"labels[{i}] must be 0/1 (int/bool), got {type(v).__name__}")

    return GroupedCandidates(problem_id=problem_id, candidates=cand_out, labels=labels_out)

