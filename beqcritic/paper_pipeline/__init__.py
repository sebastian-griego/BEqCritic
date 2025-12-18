"""
Helpers to run a paper-style pipeline around BEqCritic.

This subpackage is intentionally lightweight:
  - `clean_candidates.py`: normalize model outputs into typecheckable Lean decls
  - `typecheck_filter.py`: filter candidates by Lean typechecking (requires Lean installed)
  - `sample_candidates.py`: optional local sampling via HF Transformers
"""

