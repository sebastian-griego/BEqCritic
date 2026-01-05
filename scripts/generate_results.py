#!/usr/bin/env python3
"""
Generate `results/results.md` from a quickstart run directory.

This uses the dataset's `correct` labels as a fast proxy metric:
  selected_correct = chosen candidate has label==1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run_summarize_selection(
    *,
    candidates: Path,
    selections: Path,
    name: str,
) -> dict:
    scripts_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(scripts_dir / "summarize_selection.py"),
        "--candidates",
        str(candidates),
        "--selections",
        str(selections),
        "--name",
        str(name),
        "--max-k",
        "1",
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs/quickstart", help="Directory produced by scripts/run_quickstart.sh")
    p.add_argument("--output", default="results/results.md")
    args = p.parse_args()

    run_dir = Path(str(args.run_dir))
    cand = run_dir / "proofnetverif_test_candidates.jsonl"
    selfbleu = run_dir / "proofnetverif_test_selection_selfbleu.jsonl"
    beqcritic = run_dir / "proofnetverif_test_selection_beqcritic.jsonl"
    nlverifier = run_dir / "proofnetverif_test_selection_nlverifier.jsonl"
    if not nlverifier.exists():
        legacy = run_dir / "proofnetverif_test_selection_verifier.jsonl"
        if legacy.exists():
            nlverifier = legacy

    missing = [str(p) for p in [cand, selfbleu, beqcritic] if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing quickstart outputs:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\nRun: `bash scripts/run_quickstart.sh`"
        )

    metrics = []
    metrics.append(
        _run_summarize_selection(
            candidates=cand,
            selections=selfbleu,
            name="selfbleu",
        )
    )
    metrics.append(
        _run_summarize_selection(
            candidates=cand,
            selections=beqcritic,
            name="beqcritic",
        )
    )
    if nlverifier.exists():
        metrics.append(
            _run_summarize_selection(
                candidates=cand,
                selections=nlverifier,
                name="nlverifier",
            )
        )

    out_path = Path(str(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Results: ProofNetVerif test (selection proxy)")
    lines.append("")
    lines.append(
        "Metric: `selected_correct` = the chosen candidate has `correct==1` in `PAug/ProofNetVerif` "
        "(fast proxy for BEq+/semantic equivalence)."
    )
    lines.append("")
    lines.append(f"Inputs: `{cand}`")
    lines.append("")
    lines.append("| method | selected correct (%) | selected correct \\| any correct (%) | problems |")
    lines.append("|---|---:|---:|---:|")
    for m in metrics:
        name = m.get("name") or "-"
        sc = float(m.get("selected_correct_pct") or 0.0)
        sc_any = float(m.get("selected_correct_given_any_pct") or 0.0)
        n = int(m.get("problems") or 0)
        lines.append(f"| {name} | {sc:.1f} | {sc_any:.1f} | {n} |")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
