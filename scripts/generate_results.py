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


def _load_jsonl_map(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = obj.get("problem_id")
            if pid is None:
                raise ValueError(f"Missing problem_id at {path}:{line_no}")
            out[str(pid)] = obj
    return out


def _load_labels_map(candidates: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    with candidates.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = obj.get("problem_id")
            if pid is None:
                raise ValueError(f"Missing problem_id at {candidates}:{line_no}")
            labels = obj.get("labels")
            if not isinstance(labels, list):
                raise ValueError(f"Missing labels list for {pid} at {candidates}:{line_no}")
            out[str(pid)] = [1 if int(x) else 0 for x in labels]
    return out


def _load_choice_map(selections: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for pid, obj in _load_jsonl_map(selections).items():
        chosen_indices = obj.get("chosen_indices")
        if isinstance(chosen_indices, list) and chosen_indices:
            out[pid] = int(chosen_indices[0])
            continue
        if "chosen_index" in obj:
            out[pid] = int(obj.get("chosen_index"))
            continue
        raise ValueError(f"Missing chosen_index for {pid} in {selections}")
    return out


def _selected_correct_map(labels_map: dict[str, list[int]], choices: dict[str, int]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for pid, idx in choices.items():
        labels = labels_map.get(pid)
        if labels is None:
            continue
        if idx < 0 or idx >= len(labels):
            raise ValueError(f"chosen_index out of range for {pid}: {idx} (n={len(labels)})")
        out[pid] = bool(labels[idx])
    return out


def _pairwise_wins(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, int, int]:
    ids = sorted(set(a.keys()) & set(b.keys()))
    win = lose = tie = 0
    for pid in ids:
        ca = bool(a[pid])
        cb = bool(b[pid])
        if ca and not cb:
            win += 1
        elif cb and not ca:
            lose += 1
        else:
            tie += 1
    return win, lose, tie, len(ids)


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

    any_correct_pct = float(metrics[0].get("has_any_correct_pct") or 0.0)
    any_correct = int(metrics[0].get("has_any_correct") or 0)
    problems = int(metrics[0].get("problems") or 0)

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
    lines.append(f"Oracle ceiling (any correct): {any_correct_pct:.1f}% ({any_correct}/{problems})")
    if nlverifier.exists():
        lines.append(
            "NLVerifier provenance: `runs/verifier_v1` is documented in `REPORT.md` as trained on the "
            "local ProofNetVerif `train` split with `--eval-size 0.1` (no test split used)."
        )
    lines.append("")
    lines.append("Definitions:")
    lines.append("- `any correct (%)` is the oracle reachability of the candidate pool.")
    lines.append("- `selected correct | any correct (%)` is selector quality conditional on reachability.")
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
    if nlverifier.exists():
        lines.append("Note: `nlverifier` conditions on the natural-language statement; other methods are candidate-only.")
        lines.append("")

        labels_map = _load_labels_map(cand)
        sel_self = _selected_correct_map(labels_map, _load_choice_map(selfbleu))
        sel_beq = _selected_correct_map(labels_map, _load_choice_map(beqcritic))
        sel_nl = _selected_correct_map(labels_map, _load_choice_map(nlverifier))

        win, lose, tie, n = _pairwise_wins(sel_nl, sel_self)
        lines.append(f"Paired wins (selected_correct): nlverifier vs selfbleu = {win}/{lose}/{tie} (n={n})")
        win, lose, tie, n = _pairwise_wins(sel_nl, sel_beq)
        lines.append(f"Paired wins (selected_correct): nlverifier vs beqcritic = {win}/{lose}/{tie} (n={n})")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
