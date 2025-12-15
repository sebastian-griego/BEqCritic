"""
Cheap, purely textual structural features.

Motivation:
  - Equivalence metrics based on proof search can return false positives when a prediction
    introduces extra assumptions and then becomes trivially provable.
  - A learned scorer benefits from having cheap, explicit hints about structure.

These features do not attempt formal reasoning. They are only auxiliary tokens.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

_BINDER_RE = re.compile(r"\([^)]*:[^)]*\)")  # "(x y : T)" / "(h : P)" etc.

@dataclass(frozen=True)
class LeanStructuralFeatures:
    n_chars: int
    n_binders: int
    n_arrows: int
    n_forall: int
    n_exists: int
    n_and: int
    n_or: int
    n_eq: int
    n_prop_assumptions: int

    def to_prefix(self) -> str:
        return (
            f"[FEATS chars={self.n_chars} binders={self.n_binders} arrows={self.n_arrows} "
            f"forall={self.n_forall} exists={self.n_exists} and={self.n_and} or={self.n_or} "
            f"eq={self.n_eq} propAsm={self.n_prop_assumptions}]"
        )

def extract_features(stmt: str) -> LeanStructuralFeatures:
    n_chars = len(stmt)
    binders = _BINDER_RE.findall(stmt)
    n_binders = len(binders)
    n_arrows = stmt.count("→") + stmt.count("->")
    n_forall = stmt.count("∀")
    n_exists = stmt.count("∃")
    n_and = stmt.count("∧")
    n_or = stmt.count("∨")
    n_eq = stmt.count("=")

    # Count binder segments that look like Prop assumptions
    n_prop_assumptions = 0
    for b in binders:
        if ": Prop" in b:
            n_prop_assumptions += 1
        elif re.search(r":\s*[a-z]\w*\s*\)?", b) and "." not in b:
            n_prop_assumptions += 1

    return LeanStructuralFeatures(
        n_chars=n_chars,
        n_binders=n_binders,
        n_arrows=n_arrows,
        n_forall=n_forall,
        n_exists=n_exists,
        n_and=n_and,
        n_or=n_or,
        n_eq=n_eq,
        n_prop_assumptions=n_prop_assumptions,
    )

def featurize_pair(a: str, b: str) -> tuple[str, str]:
    fa = extract_features(a).to_prefix()
    fb = extract_features(b).to_prefix()
    return fa, fb
