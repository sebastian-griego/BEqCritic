# BEqCritic v0.2.0

A learned alternative to Lean proof-search equivalence checks.

Focus:
- decide equivalence between pairs of Lean statements quickly
- cluster autoformalization candidates by predicted equivalence and select from the largest class

Contents:
- beqcritic/inspect_dataset.py: print dataset columns and sample rows
- beqcritic/train_beq_critic.py: train the critic
- beqcritic/score_and_select.py: apply clustering and selection

Minimal training run:

python -m beqcritic.inspect_dataset --dataset PAug/ProofNetVerif --split train

python -m beqcritic.train_beq_critic \
  --dataset PAug/ProofNetVerif \
  --split train \
  --pred-key prediction \
  --ref-key reference \
  --label-key label \
  --problem-id-key problem_id \
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

python -m beqcritic.evaluate_selection \
  --candidates proofnetverif_test_candidates.jsonl \
  --selections proofnetverif_test_selection.jsonl

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
  --bootstrap 1000 \
  --report-buckets \
  --report-cand-buckets \
  --report-comp-buckets
