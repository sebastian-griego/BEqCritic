# BEqCritic

BEqCritic is a learned alternative to surface-overlap (e.g. Self-BLEU) for selecting a single Lean statement from a set
of autoformalization candidates.

Core idea:
- score candidate–candidate equivalence with a fast cross-encoder (`BeqCritic`)
- build a similarity graph, cluster, and pick a representative (e.g. medoid of the largest/cohesive cluster)
- benchmark selection strategies with uncertainty + bucketed slice reports

## Quickstart (ProofNetVerif)

Train a critic:

```bash
python -m beqcritic.train_beq_critic \
  --dataset PAug/ProofNetVerif \
  --split train \
  --pred-key lean4_prediction \
  --ref-key lean4_formalization \
  --label-key correct \
  --problem-id-key id \
  --base-model microsoft/deberta-v3-base \
  --output-dir checkpoints/beqcritic_deberta \
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
  --output proofnetverif_test_candidates.jsonl
```

Select by clustering (critic / BLEU / hybrid):

```bash
python -m beqcritic.score_and_select \
  --model checkpoints/beqcritic_deberta \
  --input proofnetverif_test_candidates.jsonl \
  --output proofnetverif_test_selection.jsonl \
  --device cuda:0 \
  --similarity critic \
  --threshold 0.5 \
  --tie-break medoid \
  --cluster-rank size_then_cohesion \
  --triangle-prune-margin 0.2 \
  --emit-stats
```

Evaluate selections:

```bash
python -m beqcritic.evaluate_selection \
  --candidates proofnetverif_test_candidates.jsonl \
  --selections proofnetverif_test_selection.jsonl
```

Benchmark a sweep (no re-scoring across thresholds/strategies):

```bash
python -m beqcritic.benchmark_selection \
  --model checkpoints/beqcritic_deberta \
  --input proofnetverif_test_candidates.jsonl \
  --device cuda:0 \
  --similarity critic \
  --thresholds 0.3,0.4,0.5,0.6 \
  --tie-breaks medoid,shortest,first \
  --cluster-ranks size_then_cohesion,size \
  --mutual-ks 0,3 \
  --triangle-prune-margins 0.0,0.2 \
  --cluster-mode support \
  --support-frac 0.7 \
  --bootstrap 1000 \
  --report-buckets \
  --report-cand-buckets \
  --report-comp-buckets
```

## More

See `README_BEQCRITIC.md` for a fuller walkthrough, tuning notes, and additional CLI examples.
