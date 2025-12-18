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

Clean train/dev/test on ProofNetVerif (avoid test leakage):

Preferred setup (uses the dataset's official splits):

1) Train on `train` and evaluate on `valid` (no overlap on `id`), and write the split ids:

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node 3 -m beqcritic.train_beq_critic \
  --dataset PAug/ProofNetVerif \
  --split train \
  --eval-split valid \
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
