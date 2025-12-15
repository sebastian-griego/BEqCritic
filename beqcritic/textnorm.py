"""
Lightweight text normalization utilities for Lean theorem/lemma statements.

Goals:
  - remove proofs and theorem names when present
  - normalize whitespace
  - keep the statement itself as close as possible to the original

This is intentionally heuristic. It does not require Lean.
"""
from __future__ import annotations
import re

_WS_RE = re.compile(r"\s+")
_LEAN_DECL_RE = re.compile(r"^\s*(theorem|lemma|example|def|axiom|structure|instance)\s+", re.IGNORECASE)

def strip_comments(s: str) -> str:
    # Remove Lean line comments
    s = re.sub(r"--.*?$", "", s, flags=re.MULTILINE)
    # Remove Lean block comments (non-nested heuristic)
    s = re.sub(r"/-.*?-/", "", s, flags=re.DOTALL)
    return s

def strip_proof(s: str) -> str:
    """
    Remove everything from the first ':=' or 'where' onwards.
    This matches most autoformalization outputs that append ':= by ...' or ':= sorry'.
    """
    for delim in [":=", "where"]:
        idx = s.find(delim)
        if idx != -1:
            return s[:idx]
    return s

def strip_decl_header(s: str) -> str:
    """
    If input is 'theorem name (args) : TYPE', drop 'theorem name' and keep '(args) : TYPE'.
    If there is no name, leave unchanged.
    """
    s0 = s.lstrip()
    m = _LEAN_DECL_RE.match(s0)
    if not m:
        return s
    rest = s0[m.end():]
    rest = rest.lstrip()
    # Remove one identifier token as the name
    rest = re.sub(r"^[A-Za-z_][A-Za-z0-9_'.]*\s*", "", rest)
    return rest

def normalize_whitespace(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = _WS_RE.sub(" ", s).strip()
    return s

def normalize_lean_statement(s: str) -> str:
    s = strip_comments(s)
    s = strip_proof(s)
    s = strip_decl_header(s)
    s = normalize_whitespace(s)
    return s
