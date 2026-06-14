# BEqCritic

BEqCritic is a learned alternative to surface-overlap (e.g. Self-BLEU) for selecting a single Lean statement from a set
of autoformalization candidates.

Core idea:
- score candidate–candidate equivalence with a fast cross-encoder (`BeqCritic`)
- build a similarity graph, cluster, and pick a representative (e.g. medoid of the largest/cohesive cluster)
- benchmark selection strategies with bucketed slice reports

## Installation

Requires Python `>=3.10`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m beqcritic.smoke
```

Optional (BEq+ paper metric; requires `lean-interact` + a working Lean toolchain):

```bash
python -m pip install -e '.[beqplus]'
```

## Quickstart (ProofNetVerif)

End-to-end (train → candidates → select → eval → A/B report):

```bash
make setup
make quickstart
make results
cat results/results.md
```

All quickstart artifacts/logs are written under `runs/quickstart/`.

Quickstart trains on a single GPU for stability. Choose the GPU with `BEQCRITIC_TRAIN_CUDA_VISIBLE_DEVICES`:

```bash
BEQCRITIC_TRAIN_CUDA_VISIBLE_DEVICES=0 make quickstart
BEQCRITIC_TRAIN_CUDA_VISIBLE_DEVICES=3 make quickstart
```

For multi-GPU DDP training, use `torchrun` (one process per GPU); see `README_BEQCRITIC.md` for examples.

Manual steps (if you want to run each stage yourself):

Train a critic (downloads the base model; set `BEQCRITIC_BASE_MODEL` to override):

```bash
python -m beqcritic.train_beq_critic \
  --dataset PAug/ProofNetVerif \
  --split valid \
  --pred-key lean4_prediction \
  --ref-key lean4_formalization \
  --label-key correct \
  --problem-id-key id \
  --base-model microsoft/deberta-v3-small \
  --output-dir runs/myrun/checkpoints/beqcritic_deberta \
  --task-mix pred_vs_ref,cand_vs_cand \
  --epochs 1 \
  --batch-size 8
```

Build grouped candidate sets (one JSON line per problem):

```bash
python -m beqcritic.make_grouped_candidates \
  --dataset PAug/ProofNetVerif \
  --split test \
  --pred-key lean4_prediction \
  --ref-key lean4_formalization \
  --label-key correct \
  --problem-id-key id \
  --output runs/myrun/proofnetverif_test_candidates.jsonl
```

Select by clustering (critic / BLEU / hybrid):

```bash
python -m beqcritic.score_and_select \
  --model runs/myrun/checkpoints/beqcritic_deberta \
  --input runs/myrun/proofnetverif_test_candidates.jsonl \
  --output runs/myrun/proofnetverif_test_selection.jsonl \
  --device cpu \
  --similarity critic \
  --threshold 0.5 \
  --tie-break medoid \
  --cluster-rank size_then_cohesion \
  --triangle-prune-margin 0.2 \
  --emit-stats
```

The selector validates the full grouped-candidate JSONL before loading a critic
or writing selection/audit outputs. Malformed rows, missing `problem_id`, empty
or non-string candidates, and label/candidate length mismatches fail early with
the physical input line number.

Evaluate selections:

```bash
python -m beqcritic.evaluate_selection \
  --candidates runs/myrun/proofnetverif_test_candidates.jsonl \
  --selections runs/myrun/proofnetverif_test_selection.jsonl
```

Metric, comparison, and BEq+ paper-pipeline CLIs reject duplicate `problem_id`
rows rather than silently keeping the last row, so accidental concatenation,
partial reruns, or corrupted resume ledgers cannot overwrite earlier records
inside reported scores.

Compare two selectors with Wilson confidence intervals and a paired exact
sign test:

```bash
python -m beqcritic.compare_selection_methods \
  --candidates runs/myrun/proofnetverif_test_candidates.jsonl \
  --selections-a runs/myrun/proofnetverif_test_selection_selfbleu.jsonl \
  --selections-b runs/myrun/proofnetverif_test_selection_beqcritic.jsonl \
  --a-name selfbleu \
  --b-name beqcritic \
  --output-md runs/myrun/selection_comparison.md
```

Build a multi-method leaderboard with pairwise sign tests, deterministic
`first` / `shortest` baselines, and optional explicit abstention rows for
selective methods:

```bash
python -m beqcritic.selection_leaderboard \
  --candidates runs/myrun/proofnetverif_test_candidates.jsonl \
  --include-baseline first \
  --include-baseline shortest \
  --selection self_bleu=runs/myrun/proofnetverif_test_selection_selfbleu.jsonl \
  --selection critic=runs/myrun/proofnetverif_test_selection_beqcritic.jsonl \
  --selection nlverifier=runs/myrun/proofnetverif_test_selection_nlverifier.jsonl \
  --selection nlverifier_abstain_p50=runs/myrun/sel_nlverifier_abstain_p50.jsonl \
  --abstention nlverifier_abstain_p50=runs/myrun/nlverifier_abstentions_p50.jsonl \
  --output-md runs/myrun/selection_leaderboard.md \
  --output-json runs/myrun/selection_leaderboard.json
```

Accepted and abstained files that share a method name are joined before
scoring, so the leaderboard reports both selected-correct rate and accepted
accuracy for the operational selective policy.

For reproducible error analysis, ask the selector to write a compact audit JSONL alongside the selections:

```bash
python -m beqcritic.score_and_select \
  --model runs/myrun/checkpoints/beqcritic_deberta \
  --input runs/myrun/proofnetverif_test_candidates.jsonl \
  --output runs/myrun/proofnetverif_test_selection.jsonl \
  --audit-output runs/myrun/proofnetverif_test_selection_audit.jsonl \
  --device cpu \
  --emit-stats
```

Each audit line records the final choice, fallback status, graph statistics, ranked clusters, top scored edges, and compact candidate snippets.

Benchmark a sweep (no re-scoring across thresholds/strategies):

```bash
python -m beqcritic.benchmark_selection \
  --model runs/myrun/checkpoints/beqcritic_deberta \
  --input runs/myrun/proofnetverif_test_candidates.jsonl \
  --device cpu \
  --similarity critic \
  --critic-pair-mode all \
  --thresholds 0.3,0.4,0.5,0.6 \
  --tie-breaks medoid,shortest,first \
  --cluster-ranks size_then_cohesion,size \
  --mutual-ks 0,3 \
  --triangle-prune-margins 0.0,0.2 \
  --cluster-mode support \
  --support-frac 0.7 \
  --report-buckets \
  --report-cand-buckets \
  --report-comp-buckets
```

Optional:

- Calibrate temperature scaling (writes `temperature.json` into the checkpoint dir):
  `python -m beqcritic.calibrate_temperature --model runs/myrun/checkpoints/beqcritic_deberta --input runs/myrun/proofnetverif_test_candidates.jsonl --device cpu`
- Self-BLEU-like consensus selection with critic similarities (global medoid, no thresholding):
  `python -m beqcritic.score_and_select --model runs/myrun/checkpoints/beqcritic_deberta --input runs/myrun/proofnetverif_test_candidates.jsonl --output runs/myrun/proofnetverif_test_selection_mbr.jsonl --select-mode global_medoid --medoid-objective mean --device cpu`
- BEq+-friendlier representative selection (pick simplest among top-k medoid candidates):
  `python -m beqcritic.score_and_select --model runs/myrun/checkpoints/beqcritic_deberta --input runs/myrun/proofnetverif_test_candidates.jsonl --output runs/myrun/proofnetverif_test_selection_simple.jsonl --medoid-simple-top-k 3 --simple-weight-chars 1.0 --simple-weight-binders 0.5 --simple-weight-prop-assumptions 0.25 --device cpu`
- Reduce scoring cost for large candidate sets (score only kNN edges):
  `python -m beqcritic.score_and_select --model runs/myrun/checkpoints/beqcritic_deberta --similarity critic --critic-pair-mode knn --knn-k 10 ...`
- Debug one problem (inspect clusters + top edges):
  `python -m beqcritic.debug_selection --input runs/myrun/proofnetverif_test_candidates.jsonl --problem-id <id> --model runs/myrun/checkpoints/beqcritic_deberta --similarity critic --device cpu`

## NLVerifier (NL->Lean reranker)

NLVerifier scores `(nl_statement, lean_statement)` pairs and selects the top candidate (NLVerifier-Select). This is
distinct from BEqCritic, which scores Lean<->Lean similarity and clusters candidates.
Note: NLVerifier conditions on the natural-language statement; selfbleu/BEqCritic are candidate-only baselines.
NLVerifier is the NL->Lean reranker (previously labeled 'verifier' in older run artifacts).

Example selection (prefer new output names containing `nlverifier`):

```bash
python -m beqcritic.verifier_select \
  --model runs/myrun/checkpoints/nlverifier_deberta \
  --dataset <hf_dataset> --split test \
  --input runs/myrun/proofnetverif_test_candidates.jsonl \
  --output runs/myrun/proofnetverif_test_selection_nlverifier.jsonl \
  --emit-scores \
  --emit-confidence \
  --calibration-json runs/myrun/nlverifier_calibration.json
```

The selector validates the grouped candidate JSONL before loading models or
writing output, so malformed rows, missing problem IDs, candidate/typecheck
shape errors, and missing NL statements fail early instead of producing partial
selection files.

Turn emitted scores into a ranking-quality and failure-analysis report:

```bash
python -m beqcritic.nlverifier_diagnostics \
  --candidates runs/myrun/proofnetverif_test_candidates.jsonl \
  --selections runs/myrun/proofnetverif_test_selection_nlverifier.jsonl \
  --output-md runs/myrun/nlverifier_diagnostics.md \
  --output-json runs/myrun/nlverifier_diagnostics.json \
  --failures-jsonl runs/myrun/nlverifier_failures.jsonl
```

Check whether emitted scores are calibrated as probabilities and fit a scalar
temperature for reporting:

```bash
python -m beqcritic.nlverifier_calibration \
  --scores runs/myrun/nlverifier_scores.jsonl \
  --fit-temperature \
  --output-md runs/myrun/nlverifier_calibration.md \
  --output-json runs/myrun/nlverifier_calibration.json
```

Measure selective-prediction behavior by abstaining on low-confidence cases:

```bash
python -m beqcritic.nlverifier_selective \
  --scores runs/myrun/nlverifier_scores.jsonl \
  --calibration-json runs/myrun/nlverifier_calibration.json \
  --confidence-key chosen_probability \
  --output-md runs/myrun/nlverifier_selective_risk.md \
  --output-json runs/myrun/nlverifier_selective_risk.json
```

Compare all emitted confidence signals for abstention:

```bash
python -m beqcritic.nlverifier_confidence_audit \
  --scores runs/myrun/nlverifier_scores.jsonl \
  --calibration-json runs/myrun/nlverifier_calibration.json \
  --output-md runs/myrun/nlverifier_confidence_audit.md \
  --output-json runs/myrun/nlverifier_confidence_audit.json
```

Choose abstention thresholds with Wilson lower-bound accuracy checks:

```bash
python -m beqcritic.nlverifier_thresholds \
  --scores runs/myrun/nlverifier_scores.jsonl \
  --calibration-json runs/myrun/nlverifier_calibration.json \
  --confidence-keys chosen_probability,probability_margin,score_margin \
  --output-md runs/myrun/nlverifier_thresholds.md \
  --output-json runs/myrun/nlverifier_thresholds.json
```

Audit the chosen threshold's leave-one-out stability before deploying it:

```bash
python -m beqcritic.nlverifier_threshold_stability \
  --scores runs/myrun/nlverifier_scores.jsonl \
  --calibration-json runs/myrun/nlverifier_calibration.json \
  --confidence-key chosen_probability \
  --target-accuracy 0.5 \
  --output-md runs/myrun/nlverifier_threshold_stability_p50.md \
  --output-json runs/myrun/nlverifier_threshold_stability_p50.json
```

Apply a recommended threshold to produce accepted selections, abstentions, and
an audit report:

```bash
python -m beqcritic.nlverifier_abstain \
  --scores runs/myrun/nlverifier_scores.jsonl \
  --calibration-json runs/myrun/nlverifier_calibration.json \
  --thresholds-json runs/myrun/nlverifier_thresholds.json \
  --confidence-key chosen_probability \
  --target-accuracy 0.5 \
  --require-certified \
  --output-accepted runs/myrun/sel_nlverifier_abstain_p50.jsonl \
  --output-abstained runs/myrun/nlverifier_abstentions_p50.jsonl \
  --output-md runs/myrun/nlverifier_abstention_policy_p50.md \
  --output-json runs/myrun/nlverifier_abstention_policy_p50.json
```

Evaluate the accepted and abstained outputs together:

```bash
python -m beqcritic.evaluate_selection \
  --candidates runs/myrun/nlverifier_scores.jsonl \
  --selections runs/myrun/sel_nlverifier_abstain_p50.jsonl \
  --abstentions runs/myrun/nlverifier_abstentions_p50.jsonl \
  --summary-json runs/myrun/metrics_nlverifier_abstain_p50.json
```

Build an inspectable casebook that joins accepted/abstained rows back to the
scored candidates:

```bash
python -m beqcritic.nlverifier_abstention_cases \
  --scores runs/myrun/nlverifier_scores.jsonl \
  --accepted runs/myrun/sel_nlverifier_abstain_p50.jsonl \
  --abstained runs/myrun/nlverifier_abstentions_p50.jsonl \
  --output-md runs/myrun/nlverifier_abstention_cases_p50.md \
  --output-json runs/myrun/nlverifier_abstention_cases_p50.json \
  --cases-jsonl runs/myrun/nlverifier_abstention_cases_p50.jsonl
```

On `results/exp_inductive`, the certified 50% Wilson-LCB threshold accepts
`36/55` problems and raises accepted-selection accuracy from full-coverage
`49.1%` to `66.7%` (`24/36`), while abstaining on `19/55`; the accepted
bucket's oracle ceiling is `25/36`, so only one accepted problem still has a
correct candidate that NLVerifier missed. The abstention-aware evaluator writes
the same operational split to `results/exp_inductive/metrics_nlverifier_abstain_p50.json`.
The confidence audit in
`results/exp_inductive/nlverifier_confidence_audit.md` shows
`chosen_probability` is the strongest selective-prediction signal: its
area-under-accuracy-coverage is `+15.2` points over full coverage, with
`69.8%` average precision and `71.8%` oracle-normalized accuracy area.
The multi-method leaderboard in `results/exp_inductive/selection_leaderboard.md`
shows full-coverage NLVerifier at `27/55` selected correct, ahead of `critic`
and `hybrid` at `18/55`, `self_bleu` at `17/55`, `first` at `13/55`, and
`shortest` at `6/55`. The certified abstention policy is also on the
coverage/accuracy frontier: it covers `36/55` problems, raises accepted
accuracy to `66.7%` (`24/36`), and abstains on `19/55` problems while giving
up three full-coverage correct selections. Paired sign tests have full
NLVerifier beating `critic` and `hybrid` by `9` wins to `0` losses
(`p = 0.00390625`) and `self_bleu` by `10` to `0` (`p = 0.00195312`).
The leave-one-out stability report in
`results/exp_inductive/nlverifier_threshold_stability_p50.md` finds three
nearby recommended thresholds (`0.5132`, `0.5413`, `0.6006`); although the exact
cutoff changes in `36/55` folds, applying each fold's cutoff back to the full
sample accepts `35` to `37` problems and keeps the accepted set at least `97.2%`
Jaccard-similar to the full-sample policy.
The casebook in `results/exp_inductive/nlverifier_abstention_cases_p50.md`
shows that the 12 accepted errors include 11 problems with no correct candidate
available, and that the abstained bucket contains 3 correct selections plus 1
miss with an available correct candidate.

Regenerate the paper-ready NLVerifier rollup after refreshing any of the
component reports. The rollup records each source path and SHA-256 digest so
reported tables can be traced to exact checked-in result bytes:

```bash
python scripts/summarize_nlverifier_paper_metrics.py \
  --results-dir results \
  --output-json results/nlverifier_paper_metrics.json \
  --output-md results/nlverifier_paper_metrics.md \
  --output-tex paper/generated/nlverifier_main_table.tex
```

Run the full local reproducibility gate used by CI:

```bash
python scripts/verify_reproducibility.py
```

With `make` available, the equivalent shortcut is:

```bash
make verify
```

Write a machine-readable command report:

```bash
python scripts/verify_reproducibility.py --report-json runs/reproducibility_report.json
```

With `make` available:

```bash
make verify-report
```

The JSON report records `schema_version`, planned and executed command counts,
per-command return codes, elapsed times, and the first failed command when the
gate stops early.

CI uploads this report as an artifact for each supported Python version.

Verify the checked-in rollup and generated table are current without rewriting
them:

```bash
python scripts/summarize_nlverifier_paper_metrics.py \
  --results-dir results \
  --output-json results/nlverifier_paper_metrics.json \
  --output-md results/nlverifier_paper_metrics.md \
  --output-tex paper/generated/nlverifier_main_table.tex \
  --check
```

With `make` available:

```bash
make paper-check
```

Verify that the source hashes embedded in the checked-in rollup still match the
current source result files:

```bash
python scripts/summarize_nlverifier_paper_metrics.py \
  --output-json results/nlverifier_paper_metrics.json \
  --verify-source-hashes
```

With `make` available:

```bash
make source-hashes
```

Prefer new output filenames containing `nlverifier` (existing `runs/` artifacts keep their original names).

## More

See `README_BEQCRITIC.md` for a fuller walkthrough, tuning notes, and additional CLI examples.

Repo layout:
- `runs/` (ignored): generated outputs (quickstart writes to `runs/quickstart/`)
- `examples/`: small sample files for inspection
- `artifacts/`: larger experiment outputs (organized; optional)

If you have candidates from another pipeline (e.g. the paper’s) and just want to swap the selection stage:

- Convert flat JSONL to grouped: `python -m beqcritic.group_candidates_jsonl --input filtered_flat.jsonl --output filtered_grouped.jsonl`
- Self-BLEU baseline: `python -m beqcritic.self_bleu_select --input filtered_grouped.jsonl --output selfbleu_selection.jsonl`
- BEqCritic selection: `python -m beqcritic.score_and_select --model <ckpt> --input filtered_grouped.jsonl --output beqcritic_selection.jsonl ...`
- BEq+ evaluation (paper metric): `python -m beqcritic.paper_pipeline.beq_plus_eval --dataset <hf_dataset> --split <split> --selections-a selfbleu_selection.jsonl --selections-b beqcritic_selection.jsonl ...` (requires `pip install -e '.[beqplus]'`)
