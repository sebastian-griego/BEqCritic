# NLVerifier paper-ready metrics

All numbers are computed with NLVerifier `runs/verifier_v1/checkpoints/nl_verifier_deberta_v3_base`.
NL-blank uses empty string; NL-const uses: "This is a fixed placeholder NL statement."

## ProofNetVerif (transductive test)

| setting | selected_correct (%) | selected_correct_given_any (%) | any_correct (%) | MRR | Hit@3 (%) | Hit@5 (%) | problems |
|---|---:|---:|---:|---:|---:|---:|---:|
| NL (full) | 62.4 | 92.5 | 67.4 | 0.955 | 98.3 | 100.0 | 178 |
| NL blank | 56.7 | 84.2 | 67.4 | 0.911 | 99.2 | 99.2 | 178 |
| NL const | 58.4 | 86.7 | 67.4 | 0.923 | 98.3 | 98.3 | 178 |

## ProofNetVerif (inductive ID-disjoint test)

| setting | selected_correct (%) | selected_correct_given_any (%) | any_correct (%) | MRR | Hit@3 (%) | Hit@5 (%) | problems |
|---|---:|---:|---:|---:|---:|---:|---:|
| NL (full) | 49.1 | 93.1 | 52.7 | 0.960 | 100.0 | 100.0 | 55 |
| NL blank | 41.8 | 79.3 | 52.7 | 0.874 | 93.1 | 100.0 | 55 |
| NL const | 45.5 | 86.2 | 52.7 | 0.908 | 93.1 | 100.0 | 55 |

## Selective prediction and abstention (inductive)

Best integrated confidence signal: `chosen_probability`.

| confidence key | mean prefix accuracy | lift over full | average precision | oracle-normalized area | best prefix accuracy | best prefix coverage |
|---|---:|---:|---:|---:|---:|---:|
| `chosen_probability` | 64.2% | +15.2 pp | 69.8% | 71.8% | 78.9% | 34.5% |
| `probability_margin` | 51.6% | +2.5 pp | 55.6% | 53.2% | 100.0% | 3.6% |
| `score_margin` | 52.4% | +3.3 pp | 56.0% | 54.4% | 100.0% | 3.6% |

Certified 50% Wilson-LCB abstention policy:

| coverage | accepted accuracy | selected correct counting abstentions | accepted selected-correct given any | accepted | abstained | threshold |
|---:|---:|---:|---:|---:|---:|---:|
| 65.5% | 66.7% (24/36) | 43.6% (24/55) | 96.0% (24/25) | 36 | 19 | 0.5413 |

Leave-one-out threshold stability:

- Unique thresholds: `3`
- Threshold range: `0.5132` to `0.6006`
- Changed threshold in `36/55` folds
- Applied full-sample accepted range: `35` to `37`
- Minimum accepted-set Jaccard: `97.2%`

Coverage/accuracy frontier:

| method | coverage | accepted accuracy | accepted |
|---|---:|---:|---:|
| `nlverifier` | 100.0% | 49.1% (27/55) | 55 |
| `nlverifier_abstain_p50` | 65.5% | 66.7% (24/36) | 36 |

## Multi-method selection leaderboard (inductive)

Best method by selected-correct count: `nlverifier`.

| method | coverage | selected correct | accepted accuracy | missed available | abstained |
|---|---:|---:|---:|---:|---:|
| `nlverifier` | 100.0% | 49.1% (27/55) | 49.1% (27/55) | 2 | 0 |
| `nlverifier_abstain_p50` | 65.5% | 43.6% (24/55) | 66.7% (24/36) | 5 | 19 |
| `critic` | 100.0% | 32.7% (18/55) | 32.7% (18/55) | 11 | 0 |
| `hybrid` | 100.0% | 32.7% (18/55) | 32.7% (18/55) | 11 | 0 |
| `self_bleu` | 100.0% | 30.9% (17/55) | 30.9% (17/55) | 12 | 0 |
| `first` | 100.0% | 23.6% (13/55) | 23.6% (13/55) | 16 | 0 |
| `shortest` | 100.0% | 10.9% (6/55) | 10.9% (6/55) | 23 | 0 |

## OOD pair classification (FormalAlign minif2f misalignment test)

Pairs: 3888 (pos=243, neg=3645)
Zero-shot accuracy @thr=0: 73.6% (pos=26.3%, neg=76.7%)
Selection top1 accuracy (per-input): 2.47% (n=243)

| misalignment type | accuracy |
|---|---:|
| `aligned` | 26.3% |
| `constant` | 75.0% |
| `variable_type` | 76.0% |
| `variable_new` | 84.7% |
| `equality` | 68.3% |
| `exponent` | 70.3% |

## Qualitative analysis
See `results/exp_inductive/nlverifier_failure_cases.md` and `results/exp_inductive/nlverifier_abstention_cases_p50.md`.

## Sources

- `proofnetverif_ablation`: `results/nlverifier_proofnetverif_ablation_metrics.json` (`sha256:04d5c913c0bedf9c5b4ada735dc37dd3ecf5fbf455cb461ec8449a4a963faf0c`)
- `confidence_audit`: `results/exp_inductive/nlverifier_confidence_audit.json` (`sha256:f7afa35ef8494194fb9eba9d9d21a9b8fe74b1e737b92496691ea599bc2b3ea4`)
- `abstention_metrics`: `results/exp_inductive/metrics_nlverifier_abstain_p50.json` (`sha256:8a959f0348657fb959f2c5d80bc54171452c9fc5d1dce1252857709491d57f59`)
- `threshold_stability`: `results/exp_inductive/nlverifier_threshold_stability_p50.json` (`sha256:55aa84d234d20fb8ee9436de3638caf945881d91c5e4d714906a7560a99f6ce7`)
- `selection_leaderboard`: `results/exp_inductive/selection_leaderboard.json` (`sha256:023a6782ae731862776e195ca4efb100c16da518c7d337a2ed7f92ca0688912e`)
- `ood_formalalign`: `results/ood_formalalign_minif2f.json` (`sha256:72ee5d2c08d2f7a2356522cb6795e7bff7864bbc00ef4698f68b8c6bd7c07433`)
- `transductive_random`: `results/exp_transductive/metrics_random.json` (`sha256:7a55827440af465a8ac8bd1ae4d3fdd5dcb92420461f367ec34b20bb2de513d9`)
- `transductive_self_bleu`: `results/exp_transductive/metrics_self_bleu.json` (`sha256:fdf6135058cf3f3093236291054271f4a3ee0f95c68c05dcd3dc1c0ea4337bbc`)
- `transductive_critic_global_medoid`: `results/exp_transductive/metrics_critic_global_medoid.json` (`sha256:437b9a048ca8255f031972d52c7744ce660ece7520aef144db8ab88eb3c0bd6d`)
- `inductive_random`: `results/exp_inductive/metrics_random.json` (`sha256:8fce19c812813e8d63b880721f7097ff882f82605dbce9a29fc13790ba5f8f33`)
- `inductive_self_bleu`: `results/exp_inductive/metrics_self_bleu.json` (`sha256:3e2c7228add279d9b8261f08a4ecb4f6805cf04253bc114c5cdb32259c087c26`)
- `inductive_critic_global_medoid`: `results/exp_inductive/metrics_critic_global_medoid.json` (`sha256:063dff3f6954b5527b11b3e27b68fbe3436098ccdde96ac56b54e3d7670426cb`)
