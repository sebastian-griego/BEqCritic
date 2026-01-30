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

## OOD pair classification (FormalAlign minif2f misalignment test)

Pairs: 3888 (pos=243, neg=3645)
Zero-shot accuracy @thr=0: 73.6% (pos=26.3%, neg=76.7%)
Selection top1 accuracy (per-input): 2.47% (n=243)

## Qualitative analysis
See `results/exp_inductive/nlverifier_failure_cases.md`.

