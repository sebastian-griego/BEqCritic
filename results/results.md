# Results: ProofNetVerif selection proxy

Metric: `selected_correct` = the chosen candidate has `correct==1` in `PAug/ProofNetVerif` (fast proxy for BEq+/semantic equivalence).

Definitions:
- `any correct (%)` is the oracle reachability of the candidate pool.
- `selected correct given any correct (%)` is selector quality conditional on reachability.

# Transductive (original split)

Inputs: `results/exp_transductive/proofnetverif_test_candidates.jsonl` (from `hf_datasets/ProofNetVerif`)

Oracle ceiling (any correct): 67.4% (120/178)
NLVerifier provenance: `runs/verifier_v1/checkpoints/nl_verifier_deberta_v3_base` (trained on local ProofNetVerif `train` with `--eval-size 0.1`).

| method | selected correct (%) | selected correct given any correct (%) | problems |
|---|---:|---:|---:|
| selfbleu | 47.2 | 70.0 | 178 |
| nlverifier | 62.4 | 92.5 | 178 |

Note: `nlverifier` conditions on the natural-language statement; other methods are candidate-only.

Paired wins (selected_correct): nlverifier vs selfbleu = 30/1/147 (n=178)

# Inductive (ID-disjoint by problem)

Inputs: `results/exp_inductive/proofnetverif_test_candidates.jsonl` (from `hf_datasets/ProofNetVerif_id_disjoint`)

Oracle ceiling (any correct): 52.7% (29/55)
NLVerifier provenance: `runs/verifier_v1/checkpoints/nl_verifier_deberta_v3_base` (same model as above).

| method | selected correct (%) | selected correct given any correct (%) | problems |
|---|---:|---:|---:|
| selfbleu | 30.9 | 58.6 | 55 |
| nlverifier | 49.1 | 93.1 | 55 |
