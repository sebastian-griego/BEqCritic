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
    bootstrap: int,
    seed: int,
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
        "--bootstrap",
        str(int(bootstrap)),
        "--seed",
        str(int(seed)),
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def _fmt_ci(ci: object) -> str:
    if not isinstance(ci, list) or len(ci) != 2:
        return "-"
    try:
        lo = float(ci[0])
        hi = float(ci[1])
    except Exception:
        return "-"
    return f"[{lo:.1f}, {hi:.1f}]"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs/quickstart", help="Directory produced by scripts/run_quickstart.sh")
    p.add_argument("--output", default="results/results.md")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = Path(str(args.run_dir))
    cand = run_dir / "proofnetverif_test_candidates.jsonl"
    selfbleu = run_dir / "proofnetverif_test_selection_selfbleu.jsonl"
    beqcritic = run_dir / "proofnetverif_test_selection_beqcritic.jsonl"

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
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
        )
    )
    metrics.append(
        _run_summarize_selection(
            candidates=cand,
            selections=beqcritic,
            name="beqcritic",
            bootstrap=int(args.bootstrap),
            seed=int(args.seed) + 1,
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
    lines.append("| method | selected correct (%) | 95% CI | selected correct \\| any correct (%) | 95% CI | problems |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in metrics:
        name = m.get("name") or "-"
        sc = float(m.get("selected_correct_pct") or 0.0)
        sc_ci = _fmt_ci(m.get("selected_correct_ci_pct"))
        sc_any = float(m.get("selected_correct_given_any_pct") or 0.0)
        sc_any_ci = _fmt_ci(m.get("selected_correct_given_any_ci_pct"))
        n = int(m.get("problems") or 0)
        lines.append(f"| {name} | {sc:.1f} | {sc_ci} | {sc_any:.1f} | {sc_any_ci} | {n} |")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
