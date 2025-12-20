# BEqCritic v0.2.0

A learned alternative to Lean proof-search equivalence checks.

Focus:
- decide equivalence between pairs of Lean statements quickly
- cluster autoformalization candidates by predicted equivalence and select from the largest class

Contents:
- beqcritic/inspect_dataset.py: print dataset columns and sample rows
- beqcritic/train_beq_critic.py: train the critic
- beqcritic/score_and_select.py: apply clustering and selection
- beqcritic/calibrate_temperature.py: fit temperature scaling and write `temperature.json` into a checkpoint dir
- beqcritic/debug_selection.py: inspect one problem (clusters + top edges) for error analysis
- beqcritic/group_candidates_jsonl.py: convert flat (row-level) JSONL into grouped candidates JSONL
- beqcritic/self_bleu_select.py: Self-BLEU style medoid selection baseline (paper-style)

Paper-style integration (swap selection only)
--------------------------------------------

If you already have *post-typecheck* candidates per problem from another pipeline (e.g. the paper’s),
the clean substitution point is selection.

1) Convert your filtered flat JSONL (one candidate per line) into grouped candidates JSONL:

python -m beqcritic.group_candidates_jsonl \
  --input filtered_candidates_flat.jsonl \
  --output filtered_candidates_grouped.jsonl \
  --problem-id-key problem_id \
  --candidate-key candidate

2) Run a Self-BLEU style baseline selector (global BLEU medoid):

python -m beqcritic.self_bleu_select \
  --input filtered_candidates_grouped.jsonl \
  --output selfbleu_selection.jsonl

3) Run BEqCritic selection (learned equivalence + clustering):

python -m beqcritic.score_and_select \
  --model checkpoints/beqcritic_deberta \
  --input filtered_candidates_grouped.jsonl \
  --output beqcritic_selection.jsonl \
  --device cuda:0 \
  --threshold 0.5 \
  --tie-break medoid \
  --cluster-rank size_then_cohesion \
  --triangle-prune-margin 0.2

Optional helpers (if you want this repo to do the paper’s clean/filter stages):
- `python -m beqcritic.paper_pipeline.clean_candidates ...` turns raw model text into typecheckable decls (`:= by sorry`)
- `python -m beqcritic.paper_pipeline.typecheck_filter ...` filters grouped candidates by invoking Lean (requires a Lean toolchain)
- `python -m beqcritic.paper_pipeline.beq_plus_eval ...` evaluates selector outputs vs a dataset reference using BEq+ (requires `pip install lean-interact`)
- `python -m beqcritic.paper_pipeline.sweep_beqplus_ab ...` runs a small grid sweep (selection + BEq+ A/B) without bash loops

Example BEq+ sweep (A fixed, B swept over alpha/threshold):

python -m beqcritic.paper_pipeline.sweep_beqplus_ab \
  --dataset PAug/ProofNetVerif --split valid \
  --candidates proofnetverif_valid_candidates_typechecked.jsonl \
  --selections-a selections_selfbleu_valid_typechecked.jsonl --a-name selfbleu \
  --model checkpoints/beqcritic_deberta_groupsplit \
  --out-dir sweeps_valid \
  --alphas 0.5,0.6,0.7,0.8,0.9 \
  --thresholds 0.1,0.2,0.3,0.4 \
  --device cuda:0 \
  --timeout-s 60 --bootstrap 2000

Minimal training run:

ProofNetVerif on the Hub (`PAug/ProofNetVerif`) is published with `valid` and `test` splits (no `train`).

python -m beqcritic.inspect_dataset --dataset PAug/ProofNetVerif --split valid

python -m beqcritic.train_beq_critic \
  --dataset PAug/ProofNetVerif \
  --split valid \
  --pred-key lean4_prediction \
  --ref-key lean4_formalization \
  --label-key correct \
  --problem-id-key id \
  --base-model microsoft/deberta-v3-base \
  --output-dir checkpoints/beqcritic_deberta \
  --task-mix pred_vs_ref,cand_vs_cand \
  --epochs 1 \
  --batch-size 8

Build grouped candidates JSONL (for `score_and_select`) and evaluate selection:

python -m beqcritic.make_grouped_candidates \
  --dataset PAug/ProofNetVerif \
  --split test \
  --pred-key lean4_prediction \
  --ref-key lean4_formalization \
  --label-key correct \
  --problem-id-key id \
  --output proofnetverif_test_candidates.jsonl

python -m beqcritic.score_and_select \
  --model checkpoints/beqcritic_deberta \
  --input proofnetverif_test_candidates.jsonl \
  --output proofnetverif_test_selection.jsonl \
  --device cuda:0 \
  --threshold 0.5 \
  --tie-break medoid \
  --cluster-rank size_then_cohesion \
  --triangle-prune-margin 0.2 \
  --emit-stats

Optional: calibrate temperature scaling (improves threshold stability; auto-loaded when present):

python -m beqcritic.calibrate_temperature \
  --model checkpoints/beqcritic_deberta \
  --input proofnetverif_valid_train_candidates_hard_v2.jsonl \
  --device cuda:0

Optional: reduce O(n^2) scoring cost for large candidate sets (score only kNN edges):

python -m beqcritic.score_and_select \
  --model checkpoints/beqcritic_deberta \
  --input proofnetverif_test_candidates.jsonl \
  --output proofnetverif_test_selection_knn.jsonl \
  --device cuda:0 \
  --similarity critic \
  --critic-pair-mode knn \
  --knn-k 10 \
  --threshold 0.5 \
  --tie-break medoid

Alternative clustering mode (denser, bridge-resistant):

python -m beqcritic.score_and_select \
  --model checkpoints/beqcritic_deberta \
  --input proofnetverif_test_candidates.jsonl \
  --output proofnetverif_test_selection_support.jsonl \
  --threshold 0.5 \
  --cluster-mode support \
  --support-frac 0.7 \
  --tie-break medoid \
  --cluster-rank size_then_cohesion \
  --triangle-prune-margin 0.2

BEq+-friendlier representative selection (pick the simplest member among the top-k by medoid centrality):

python -m beqcritic.score_and_select \
  --model checkpoints/beqcritic_deberta \
  --input proofnetverif_test_candidates.jsonl \
  --output proofnetverif_test_selection_support_simple.jsonl \
  --threshold 0.5 \
  --cluster-mode support \
  --support-frac 0.7 \
  --tie-break medoid \
  --medoid-simple-top-k 3 \
  --simple-weight-chars 1.0 \
  --simple-weight-binders 0.5 \
  --simple-weight-prop-assumptions 0.25 \
  --triangle-prune-margin 0.2

Self-BLEU-like consensus selection (global medoid over critic similarities; no threshold graph):

python -m beqcritic.score_and_select \
  --model checkpoints/beqcritic_deberta \
  --input proofnetverif_test_candidates.jsonl \
  --output proofnetverif_test_selection_mbr.jsonl \
  --select-mode global_medoid \
  --medoid-objective mean \
  --device cuda:0

python -m beqcritic.evaluate_selection \
  --candidates proofnetverif_test_candidates.jsonl \
  --selections proofnetverif_test_selection.jsonl

Emit/evaluate top-k selections (to reduce verifier compute):

python -m beqcritic.score_and_select \
  --model checkpoints/beqcritic_deberta \
  --input proofnetverif_test_candidates.jsonl \
  --output proofnetverif_test_selection_topk.jsonl \
  --device cuda:0 \
  --cluster-mode support \
  --support-frac 0.6 \
  --threshold 0.3 \
  --mutual-k 3 \
  --triangle-prune-margin 0.2 \
  --cluster-rank size_times_cohesion \
  --tie-break medoid \
  --fallback bleu_medoid --fallback-min-component-size 2 \
  --top-k 5

python -m beqcritic.evaluate_selection \
  --candidates proofnetverif_test_candidates.jsonl \
  --selections proofnetverif_test_selection_topk.jsonl \
  --max-k 5

Benchmark multiple thresholds and selection strategies without re-scoring:

python -m beqcritic.benchmark_selection \
  --model checkpoints/beqcritic_deberta \
  --input proofnetverif_test_candidates.jsonl \
  --device cuda:0 \
  --thresholds 0.3,0.4,0.5,0.6,0.7 \
  --tie-breaks medoid,shortest,first \
  --cluster-ranks size_then_cohesion,size \
  --mutual-ks 0,3 \
  --triangle-prune-margins 0.0,0.2 \
  --cluster-mode support --support-frac 0.7 \
  --bootstrap 1000 \
  --report-buckets \
  --report-cand-buckets \
  --report-comp-buckets

To sweep support strictness without re-scoring, use `--support-fracs`:

python -m beqcritic.benchmark_selection \
  --model checkpoints/beqcritic_deberta \
  --input proofnetverif_test_candidates.jsonl \
  --thresholds 0.3 \
  --tie-breaks medoid \
  --cluster-ranks size_times_cohesion \
  --mutual-ks 3 \
  --triangle-prune-margins 0.2 \
  --cluster-mode support --support-fracs 0.6,0.7,0.8

Optional: fallback strategies (used when the chosen cluster looks weak).

- `--fallback bleu_medoid`: use self-BLEU medoid as a baseline fallback
- `--fallback critic_medoid`: use the critic score matrix to pick the global medoid (no BLEU)
- `--fallback critic_knn_medoid --fallback-knn-k 3`: critic-only fallback using mean top-k similarity (more local than global medoid)

Optional: ensemble selection (critic-only).

`benchmark_selection` can ensemble over the provided hyperparameter grid and vote:

python -m beqcritic.benchmark_selection \
  --model checkpoints/beqcritic_deberta \
  --input proofnetverif_test_candidates.jsonl \
  --thresholds 0.2,0.3,0.4,0.5 \
  --tie-breaks medoid \
  --cluster-ranks size_times_cohesion,size_then_cohesion \
  --mutual-ks 0,3 \
  --triangle-prune-margins 0.0,0.2 \
  --cluster-mode support --support-frac 0.7 \
  --ensemble --ensemble-vote majority \
  --top-strategies 10

Clean train/dev/test on ProofNetVerif (avoid test leakage):

Preferred setup (Hub dataset has `valid` + `test` only):

Train on `valid` (with a held-out `--eval-size` group split), then report selection quality on `test`.

1) Train on `valid` and write the split ids:

python -m beqcritic.train_beq_critic \
  --dataset PAug/ProofNetVerif \
  --split valid \
  --eval-size 0.1 \
  --pred-key lean4_prediction \
  --ref-key lean4_formalization \
  --label-key correct \
  --problem-id-key id \
  --task-mix pred_vs_ref,cand_vs_cand \
  --base-model microsoft/deberta-v3-base \
  --output-dir checkpoints/beqcritic_deberta_trainsplit_hard \
  --max-length 512 \
  --max-pos-per-problem 64 \
  --max-neg-per-problem 64 \
  --cand-pos-sampling hard \
  --cand-neg-sampling hard \
  --epochs 1 \
  --batch-size 8 \
  --grad-accum 1 \
  --bf16 \
  --write-split-ids

2) Tune selection hyperparameters on `proofnetverif_valid_candidates.jsonl`, then report on `proofnetverif_test_candidates.jsonl`.

If you have an offline dataset with an explicit `train` split, you can instead pass `--split train --eval-split valid`
to avoid using `--eval-size`.

Alternative setup (when you only have a single split available locally):

1) Train on `valid` with a held-out dev subset (group split by problem_id), and write the split ids:

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node 3 -m beqcritic.train_beq_critic \
  --dataset PAug/ProofNetVerif \
  --split valid \
  --pred-key lean4_prediction \
  --ref-key lean4_formalization \
  --label-key correct \
  --problem-id-key id \
  --task-mix pred_vs_ref,cand_vs_cand \
  --base-model microsoft/deberta-v3-base \
  --output-dir checkpoints/beqcritic_deberta_validsplit \
  --max-length 512 \
  --max-pos-per-problem 64 \
  --max-neg-per-problem 64 \
  --cand-pos-sampling hard \
  --cand-neg-sampling hard \
  --epochs 1 \
  --batch-size 8 \
  --grad-accum 1 \
  --bf16 \
  --eval-size 0.2 \
  --write-split-ids

2) Split the grouped candidates JSONL using those ids:

python -m beqcritic.filter_grouped_candidates \
  --input proofnetverif_valid_candidates.jsonl \
  --problem-ids-file checkpoints/beqcritic_deberta_validsplit/eval_problem_ids.txt \
  --output proofnetverif_valid_dev_candidates.jsonl

python -m beqcritic.filter_grouped_candidates \
  --input proofnetverif_valid_candidates.jsonl \
  --problem-ids-file checkpoints/beqcritic_deberta_validsplit/eval_problem_ids.txt \
  --invert \
  --output proofnetverif_valid_train_candidates.jsonl

3) Tune selection hyperparameters on `proofnetverif_valid_dev_candidates.jsonl`, then report on `proofnetverif_test_candidates.jsonl`.
