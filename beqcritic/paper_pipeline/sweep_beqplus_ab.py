"""
CLI: sweep BEqCritic selection hyperparameters and evaluate with BEq+ as a paired A/B.

This is a convenience wrapper around:
  - `python -m beqcritic.score_and_select` (produce B selections)
  - `python -m beqcritic.paper_pipeline.beq_plus_eval` (paired A/B evaluation)

Intended use (paper-style): keep candidate generation/cleaning/typecheck filtering fixed,
then swap only the selection rule and evaluate the resulting chosen statements with BEq+.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def _parse_csv_floats(s: str) -> list[float]:
    xs: list[float] = []
    for part in str(s).split(","):
        p = part.strip()
        if not p:
            continue
        xs.append(float(p))
    if not xs:
        raise ValueError("Expected a non-empty comma-separated float list")
    return xs


def _float_tag(x: float) -> str:
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s.replace("-", "m").replace(".", "p")


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="PAug/ProofNetVerif")
    p.add_argument("--split", type=str, default="valid")

    p.add_argument("--candidates", type=str, required=True, help="Grouped candidates JSONL (post-clean, ideally post-typecheck).")
    p.add_argument("--selections-a", type=str, required=True, help="Baseline selections JSONL (e.g. Self-BLEU).")
    p.add_argument("--a-name", type=str, default="selfbleu")

    p.add_argument("--model", type=str, required=True, help="Checkpoint dir for `beqcritic.score_and_select`.")
    p.add_argument("--out-dir", type=str, required=True)

    p.add_argument("--alphas", type=str, default="0.5,0.6,0.7,0.8,0.9", help="Comma-separated hybrid alphas.")
    p.add_argument("--thresholds", type=str, default="0.1,0.2,0.3,0.4", help="Comma-separated thresholds.")

    # Selection args (passed to score_and_select).
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--similarity", type=str, default="hybrid", choices=["critic", "bleu", "hybrid"])
    p.add_argument("--cluster-mode", type=str, default="support", choices=["components", "support"])
    p.add_argument("--support-frac", type=float, default=0.7)
    p.add_argument("--triangle-prune-margin", type=float, default=0.2)
    p.add_argument("--tie-break", type=str, default="medoid", choices=["medoid", "shortest", "first"])

    p.add_argument("--medoid-simple-top-k", type=int, default=5)
    p.add_argument("--medoid-simple-max-drop", type=float, default=0.1)
    p.add_argument("--simple-weight-chars", type=float, default=1.0)
    p.add_argument("--simple-weight-binders", type=float, default=0.5)
    p.add_argument("--simple-weight-prop-assumptions", type=float, default=0.25)

    # BEq+ eval args (passed to beq_plus_eval).
    p.add_argument("--lean-version", type=str, default="v4.8.0")
    p.add_argument("--timeout-s", type=int, default=60)
    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=2000)

    p.add_argument("--dataset-id-key", type=str, default="id")
    p.add_argument("--dataset-ref-key", type=str, default="lean4_formalization")
    p.add_argument("--dataset-header-key", type=str, default="lean4_src_header")
    p.add_argument("--selection-problem-id-key", type=str, default="problem_id")
    p.add_argument("--selection-chosen-key", type=str, default="chosen")

    p.add_argument("--b-name-prefix", type=str, default="beqcritic")
    p.add_argument("--skip-existing", action="store_true", help="Skip configs where output JSONL already exists.")
    p.add_argument("--dry-run", action="store_true", help="Print commands but do not execute.")

    args = p.parse_args()

    alphas = _parse_csv_floats(args.alphas)
    thresholds = _parse_csv_floats(args.thresholds)
    if str(args.similarity) != "hybrid" and len(alphas) > 1:
        raise ValueError("--alphas only applies to --similarity=hybrid; pass a single alpha or use --similarity=hybrid.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = Path(args.candidates)
    selections_a = Path(args.selections_a)

    index_path = out_dir / "index.json"
    results: list[dict] = []

    for alpha, thr in itertools.product(alphas, thresholds):
        tag = f"a{_float_tag(alpha)}_t{_float_tag(thr)}"
        sel_b = out_dir / f"selections_{args.b_name_prefix}_{tag}.jsonl"
        ab_out = out_dir / f"beqplus_ab_{tag}.jsonl"

        if args.skip_existing and ab_out.exists():
            results.append(
                {
                    "alpha": float(alpha),
                    "threshold": float(thr),
                    "tag": tag,
                    "selections_b": str(sel_b),
                    "ab_jsonl": str(ab_out),
                    "skipped": True,
                }
            )
            continue

        cmd_select = [
            sys.executable,
            "-m",
            "beqcritic.score_and_select",
            "--model",
            str(args.model),
            "--input",
            str(candidates),
            "--output",
            str(sel_b),
            "--device",
            str(args.device),
            "--similarity",
            str(args.similarity),
            "--threshold",
            str(float(thr)),
            "--cluster-mode",
            str(args.cluster_mode),
            "--support-frac",
            str(float(args.support_frac)),
            "--triangle-prune-margin",
            str(float(args.triangle_prune_margin)),
            "--tie-break",
            str(args.tie_break),
            "--medoid-simple-top-k",
            str(int(args.medoid_simple_top_k)),
            "--medoid-simple-max-drop",
            str(float(args.medoid_simple_max_drop)),
            "--simple-weight-chars",
            str(float(args.simple_weight_chars)),
            "--simple-weight-binders",
            str(float(args.simple_weight_binders)),
            "--simple-weight-prop-assumptions",
            str(float(args.simple_weight_prop_assumptions)),
        ]
        if str(args.similarity) == "hybrid":
            cmd_select.extend(["--hybrid-alpha", str(float(alpha))])

        b_name = f"{args.b_name_prefix}_{tag}"
        cmd_eval = [
            sys.executable,
            "-m",
            "beqcritic.paper_pipeline.beq_plus_eval",
            "--dataset",
            str(args.dataset),
            "--split",
            str(args.split),
            "--dataset-id-key",
            str(args.dataset_id_key),
            "--dataset-ref-key",
            str(args.dataset_ref_key),
            "--dataset-header-key",
            str(args.dataset_header_key),
            "--selections-a",
            str(selections_a),
            "--a-name",
            str(args.a_name),
            "--selections-b",
            str(sel_b),
            "--b-name",
            str(b_name),
            "--selection-problem-id-key",
            str(args.selection_problem_id_key),
            "--selection-chosen-key",
            str(args.selection_chosen_key),
            "--lean-version",
            str(args.lean_version),
            "--timeout-s",
            str(int(args.timeout_s)),
            "--max-problems",
            str(int(args.max_problems)),
            "--shuffle-seed",
            str(int(args.shuffle_seed)),
            "--bootstrap",
            str(int(args.bootstrap)),
            "--output-jsonl",
            str(ab_out),
        ]

        if args.dry_run:
            print(" ".join(cmd_select))
            print(" ".join(cmd_eval))
        else:
            subprocess.run(cmd_select, check=True)
            subprocess.run(cmd_eval, check=True)

        results.append(
            {
                "alpha": float(alpha),
                "threshold": float(thr),
                "tag": tag,
                "selections_b": str(sel_b),
                "ab_jsonl": str(ab_out),
                "skipped": False,
            }
        )
        index_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    if not results:
        raise SystemExit("No runs executed (empty sweep?)")
    index_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote sweep index: {index_path}")


if __name__ == "__main__":
    main()

