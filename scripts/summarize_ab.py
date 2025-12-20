#!/usr/bin/env python3
"""
Summarize a BEq+ A/B JSONL results file (as produced by beqcritic.paper_pipeline.beq_plus_eval).

Input format (one JSON object per line):
  {"problem_id": "...", "a_ok": true/false, "a_name": "...", "b_ok": true/false, "b_name": "..."}

Outputs:
  - counts for (a_only, b_only, both, neither)
  - overall success rates for A and B
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


def _as_bool(x: object) -> bool:
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, int):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "t", "yes", "y"}:
            return True
        if s in {"0", "false", "f", "no", "n"}:
            return False
    raise ValueError(f"Cannot interpret bool value: {x!r}")


@dataclass(frozen=True)
class _Row:
    a_ok: bool
    b_ok: bool | None
    a_name: str
    b_name: str


def _load_rows(path: str) -> list[_Row]:
    rows: list[_Row] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if "a_ok" not in obj:
                raise ValueError(f"Missing a_ok at {path}:{line_no}")
            a_ok = _as_bool(obj["a_ok"])
            b_ok = _as_bool(obj["b_ok"]) if "b_ok" in obj else None
            rows.append(
                _Row(
                    a_ok=a_ok,
                    b_ok=b_ok,
                    a_name=str(obj.get("a_name") or "A"),
                    b_name=str(obj.get("b_name") or "B"),
                )
            )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to beqplus_ab_results_*.jsonl")
    p.add_argument("--output-json", default="", help="Optional path to write JSON summary")
    args = p.parse_args()

    rows = _load_rows(str(args.input))
    n = len(rows)

    a_name = next((r.a_name for r in rows if r.a_name), "A")
    b_name = next((r.b_name for r in rows if r.b_name), "B")
    has_b = any(r.b_ok is not None for r in rows)

    a_hits = sum(1 for r in rows if r.a_ok)
    out: dict[str, object] = {
        "problems": int(n),
        "a": {"name": a_name, "hits": int(a_hits), "rate": float(a_hits) / max(1, n)},
    }

    if not has_b:
        text = [
            f"problems: {n}",
            f"{a_name}: {a_hits}/{n} = {a_hits/max(1,n):.3f}",
        ]
        print("\n".join(text))
    else:
        b_hits = sum(1 for r in rows if bool(r.b_ok))
        both = sum(1 for r in rows if r.a_ok and bool(r.b_ok))
        a_only = sum(1 for r in rows if r.a_ok and not bool(r.b_ok))
        b_only = sum(1 for r in rows if (not r.a_ok) and bool(r.b_ok))
        neither = n - both - a_only - b_only
        out.update(
            {
                "b": {"name": b_name, "hits": int(b_hits), "rate": float(b_hits) / max(1, n)},
                "contingency": {
                    "both": int(both),
                    "a_only": int(a_only),
                    "b_only": int(b_only),
                    "neither": int(neither),
                },
            }
        )

        text = [
            f"problems: {n}",
            f"{a_name}: {a_hits}/{n} = {a_hits/max(1,n):.3f}",
            f"{b_name}: {b_hits}/{n} = {b_hits/max(1,n):.3f}",
            f"both: {both}",
            f"{a_name}_only: {a_only}",
            f"{b_name}_only: {b_only}",
            f"neither: {neither}",
        ]
        print("\n".join(text))

    if str(args.output_json).strip():
        with open(str(args.output_json), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()

