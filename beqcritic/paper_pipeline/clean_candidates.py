"""
CLI: clean model outputs into typecheckable Lean declarations.

This is meant to mirror the paper pipeline's "cleaning pass":
  - strip proofs / trailing code (`:= ...`, `where ...`)
  - normalize whitespace
  - avoid theorem-name clashes by renaming candidates uniquely
  - append a dummy `sorry` proof so Lean checks the statement type only

Supports both flat and grouped JSONL inputs:
  - flat:   {"problem_id": "...", "candidate": "..."}
  - grouped:{"problem_id": "...", "candidates": ["...", ...]}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

from ..textnorm import normalize_whitespace, strip_comments, strip_decl_header, strip_proof


_FENCE_RE = re.compile(r"```(?:lean)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def _unwrap_fenced_code(text: str) -> str:
    m = _FENCE_RE.search(text)
    if not m:
        return text
    inner = m.group(1)
    return inner.strip()


def _sanitize_ident(s: str) -> str:
    s2 = re.sub(r"[^A-Za-z0-9_]", "_", str(s))
    s2 = re.sub(r"_+", "_", s2).strip("_")
    return s2 or "x"


def _clean_signature(raw: str) -> str:
    s = _unwrap_fenced_code("" if raw is None else str(raw))
    s = strip_comments(s)
    s = strip_proof(s)
    s = strip_decl_header(s)
    s = normalize_whitespace(s)
    return s


def _to_sorry_decl(signature: str, name: str, decl_kind: str = "theorem") -> str:
    sig = signature.strip()
    if not sig:
        return ""
    if sig.startswith(("(", "{", "[", ":")):
        head = sig
    else:
        head = f": {sig}"
    return f"{decl_kind} {name} {head} := by\n  sorry"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--input-format", type=str, default="auto", choices=["auto", "flat", "grouped"])
    p.add_argument("--problem-id-key", type=str, default="problem_id")
    p.add_argument("--candidate-key", type=str, default="candidate", help="Used for --input-format=flat")
    p.add_argument("--candidates-key", type=str, default="candidates", help="Used for --input-format=grouped")
    p.add_argument("--decl-kind", type=str, default="theorem")
    p.add_argument("--name-prefix", type=str, default="cand")
    p.add_argument("--drop-empty", action="store_true", help="Drop candidates that clean to empty signatures")
    args = p.parse_args()

    counts: defaultdict[str, int] = defaultdict(int)

    def _next_name(pid: str) -> str:
        i = counts[pid]
        counts[pid] += 1
        return f"{_sanitize_ident(args.name_prefix)}_{_sanitize_ident(pid)}_{i}"

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            if args.problem_id_key not in obj:
                raise ValueError(f"Missing {args.problem_id_key!r} in input row: {obj}")
            pid = str(obj[args.problem_id_key])

            fmt = str(args.input_format)
            if fmt == "auto":
                fmt = "grouped" if isinstance(obj.get(args.candidates_key), list) else "flat"

            if fmt == "flat":
                raw = obj.get(args.candidate_key)
                sig = _clean_signature("" if raw is None else str(raw))
                name = _next_name(pid)
                decl = _to_sorry_decl(sig, name=name, decl_kind=str(args.decl_kind))
                if args.drop_empty and not decl.strip():
                    continue
                obj[args.candidate_key] = decl
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            if fmt == "grouped":
                cands = obj.get(args.candidates_key) or []
                if not isinstance(cands, list):
                    raise ValueError(f"Expected {args.candidates_key!r} to be a list in grouped input: {obj}")
                out_cands: list[str] = []
                for raw in cands:
                    sig = _clean_signature("" if raw is None else str(raw))
                    name = _next_name(pid)
                    decl = _to_sorry_decl(sig, name=name, decl_kind=str(args.decl_kind))
                    if args.drop_empty and not decl.strip():
                        continue
                    out_cands.append(decl)
                obj[args.candidates_key] = out_cands
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            raise ValueError(f"Unknown input_format={args.input_format!r}")


if __name__ == "__main__":
    main()

