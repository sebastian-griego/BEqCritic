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

- `proofnetverif_ablation`: `results/nlverifier_proofnetverif_ablation_metrics.json` (`sha256:121e8cdd4ceb7bae5c44d774b57602913db3b64e7ac9f74f304fded489dde0d5`)
- `confidence_audit`: `results/exp_inductive/nlverifier_confidence_audit.json` (`sha256:190498add7f603085863e267b500dd900c76e5c9898125179f5063f8cf63b6d4`)
- `abstention_metrics`: `results/exp_inductive/metrics_nlverifier_abstain_p50.json` (`sha256:e32b9a0b7fe9a76e8aacba3c78e2464ce3a36bc4fd97b5e677c6b972afec647c`)
- `threshold_stability`: `results/exp_inductive/nlverifier_threshold_stability_p50.json` (`sha256:f8c88881ec12d49fb1b03bbaabe3f7039cede7dfb69179d98a2d29341d792770`)
- `selection_leaderboard`: `results/exp_inductive/selection_leaderboard.json` (`sha256:846ec10df6d2c31f97a32b89a674cf851e8631942456822a462d6ad3741b1e0a`)
- `ood_formalalign`: `results/ood_formalalign_minif2f.json` (`sha256:d60b88ab1372aedbcbef016c13391e52b6b5c80dcc03b185a1cfefc5c538b61c`)
- `transductive_random`: `results/exp_transductive/metrics_random.json` (`sha256:03ca128c9857d7bed98f8fef95be349afb0e55697a5f1c2fdd8dbe016a92a932`)
- `transductive_self_bleu`: `results/exp_transductive/metrics_self_bleu.json` (`sha256:3355af8500c8ae78e19f4de2d85cf47863e483e5965633c6dda3f04a436fd86d`)
- `transductive_critic_global_medoid`: `results/exp_transductive/metrics_critic_global_medoid.json` (`sha256:3686591739ea49e62ad20ff43619ee17e1d89022f89f143a669251c1e685974a`)
- `inductive_random`: `results/exp_inductive/metrics_random.json` (`sha256:3cc0d8f01b5a95e45b092346dd9daae6f41f4012b67bd0ded91b3de2a179a435`)
- `inductive_self_bleu`: `results/exp_inductive/metrics_self_bleu.json` (`sha256:ab29bf79ad44ca3c0b4fa4cd35f744c87f528b5211d315b8e435ce8c0ff57521`)
- `inductive_critic_global_medoid`: `results/exp_inductive/metrics_critic_global_medoid.json` (`sha256:ca8d965fec4e39dc2d97dafdc85f8491a26de4d3c02536db742158a6d9ea270d`)
