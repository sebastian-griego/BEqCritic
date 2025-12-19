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
  --critic-pair-mode all \
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

Optional:

- Calibrate temperature scaling (writes `temperature.json` into the checkpoint dir):
  `python -m beqcritic.calibrate_temperature --model checkpoints/beqcritic_deberta --input proofnetverif_valid_train_candidates_hard_v2.jsonl --device cuda:0`
- Self-BLEU-like consensus selection with critic similarities (global medoid, no thresholding):
  `python -m beqcritic.score_and_select --model checkpoints/beqcritic_deberta --input proofnetverif_test_candidates.jsonl --output proofnetverif_test_selection_mbr.jsonl --select-mode global_medoid --medoid-objective mean --device cuda:0`
- BEq+-friendlier representative selection (pick simplest among top-k medoid candidates):
  `python -m beqcritic.score_and_select --model checkpoints/beqcritic_deberta --input proofnetverif_test_candidates.jsonl --output proofnetverif_test_selection_simple.jsonl --medoid-simple-top-k 3 --simple-weight-chars 1.0 --simple-weight-binders 0.5 --simple-weight-prop-assumptions 0.25 --device cuda:0`
- Reduce scoring cost for large candidate sets (score only kNN edges):
  `python -m beqcritic.score_and_select --model checkpoints/beqcritic_deberta --similarity critic --critic-pair-mode knn --knn-k 10 ...`
- Debug one problem (inspect clusters + top edges):
  `python -m beqcritic.debug_selection --input proofnetverif_test_candidates.jsonl --problem-id <id> --model checkpoints/beqcritic_deberta --similarity critic --device cuda:0`

## More

See `README_BEQCRITIC.md` for a fuller walkthrough, tuning notes, and additional CLI examples.

If you have candidates from another pipeline (e.g. the paper’s) and just want to swap the selection stage:

- Convert flat JSONL to grouped: `python -m beqcritic.group_candidates_jsonl --input filtered_flat.jsonl --output filtered_grouped.jsonl`
- Self-BLEU baseline: `python -m beqcritic.self_bleu_select --input filtered_grouped.jsonl --output selfbleu_selection.jsonl`
- BEqCritic selection: `python -m beqcritic.score_and_select --model <ckpt> --input filtered_grouped.jsonl --output beqcritic_selection.jsonl ...`
- BEq+ evaluation (paper metric): `python -m beqcritic.paper_pipeline.beq_plus_eval --dataset <hf_dataset> --split <split> --selections-a selfbleu_selection.jsonl --selections-b beqcritic_selection.jsonl ...` (requires `pip install lean-interact`)
