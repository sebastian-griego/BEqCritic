# REPORT: ProofNetVerif selection quality

## What “good” means

Task: for each ProofNetVerif problem, pick **one** Lean statement from a set of autoformalization candidates.

Primary metric (proxy for BEq+/semantic equivalence): **Selected correct (%)**, i.e. the fraction of problems where
the chosen candidate has `correct=1` in `PAug/ProofNetVerif`.

Why a proxy: the paper’s BEq+ metric requires `lean-interact` plus a working Lean toolchain (downloads/builds Mathlib).
This report uses the dataset’s correctness labels for a fast, reproducible end-to-end comparison.

## Reproduce

From a clean checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .

# end-to-end: train → candidates → select → eval → A/B report
bash scripts/run_quickstart.sh
cat runs/quickstart/ab_metrics.md
```

All artifacts/logs are written under `runs/quickstart/` (see `runs/quickstart/*.log` and `runs/quickstart/timing.txt`).

## Results (ProofNetVerif test)

Run config:
- Dataset: `PAug/ProofNetVerif`
- Train split: `valid` (seed `0`, `--epochs 1`, `--batch-size 8`)
- Eval split: `test` (178 problems, avg 8.16 candidates/problem)
- BEqCritic selection: `--threshold 0.5 --tie-break medoid --cluster-rank size_then_cohesion --triangle-prune-margin 0.2`

From `runs/quickstart/ab_metrics.md`:

| method | selected correct (%) | 95% CI | selected correct \| any correct (%) | 95% CI | selection time (s) | pairwise comps |
|---|---:|---:|---:|---:|---:|---:|
| selfbleu | 47.2 | [39.9, 54.5] | 70.0 | [61.7, 78.3] | 5 | 6638 |
| beqcritic | 43.8 | [36.5, 51.1] | 65.0 | [56.7, 73.3] | 27 | 6638 |
| beqcritic - selfbleu | -3.4 | [-7.3, +0.0] | - | - | - | - |

Cost drivers:
- Both methods score all candidate pairs on this split: `∑_problems n(n-1)/2 = 6638`.
- BEqCritic’s per-pair cost is dominated by cross-encoder inference; Self-BLEU is dominated by tokenization + n-gram overlap.

## Failure modes observed

Representative examples (from `runs/quickstart`):

- False equivalence edges can create a larger *incorrect* cluster, causing size-based ranking to pick the wrong component.
  Example: `Rudin|exercise_1_18b` (BEqCritic picks an incorrect 3-node component while a 2-node correct component exists).
- Missed equivalence edges can fragment correct candidates into small components that lose against a larger noisy component.
- Surface-overlap baselines fail when the lexically “central” candidate is wrong (e.g., typeclass/structure mismatches),
  even if a small set of correct candidates is present.

## Recommendations

- **Default threshold:** start with `--threshold 0.5` and tune on a held-out dev split (or `--write-split-ids` + filter).
- **Triangle pruning / mutual-k:** keep `--triangle-prune-margin 0.2`; consider `--mutual-k 3` when you see “bridge” errors.
- **Calibration:** `beqcritic.calibrate_temperature` is cheap to run, but in this quickstart it learned `T≈1.0` (no change).
- **Scaling to larger candidate sets:** use `--critic-pair-mode knn --knn-k 10` to avoid O(n²) scoring.

## Notes on licensing

This repo does not include a `LICENSE` file; redistribution/derivative-use terms are unclear.

