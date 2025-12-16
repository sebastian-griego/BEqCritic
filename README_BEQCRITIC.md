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
