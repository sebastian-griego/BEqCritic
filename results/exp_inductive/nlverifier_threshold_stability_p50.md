# NLVerifier Threshold Stability

- Problems: 55
- Any-correct ceiling: 52.7% (29/55)
- Full-coverage selected accuracy: 49.1% (27/55)
- Confidence key: `chosen_probability`
- Target accuracy: 50.0%
- Minimum accepted examples: `5`

## Full Recommendation

| certified? | threshold | accepted | coverage | accuracy | 95% LCB |
|---:|---:|---:|---:|---:|---:|
| yes | 0.5413 | 36 | 65.5% | 66.7% | 50.3% |

## Leave-One-Out Summary

- Resamples: 55
- Unique thresholds: 3
- Threshold changed vs full: 36/55 (65.5%)
- Threshold range: `0.5132` to `0.6006`
- Train accepted range: 34 to 36
- Applied-to-full accepted range: 35 to 37
- Minimum accepted-set Jaccard vs full: 97.2%
- Held-out accepted accuracy: 66.7% (24/36; accepted 36, rejected 19)

## Threshold Values

| threshold | resamples |
|---:|---:|
| 0.6006 | 24 |
| 0.5413 | 19 |
| 0.5132 | 12 |

## Most Sensitive Omissions

| omitted problem | correct? | confidence | LOO threshold | delta | full accepted | Jaccard | held out accepted? |
|---|---:|---:|---:|---:|---:|---:|---:|
| Shakarchi\|exercise_1_13c | yes | 0.7783 | 0.6006 | +0.0593 | 35 | 97.2% | yes |
| Rudin\|exercise_4_2a | yes | 0.9362 | 0.6006 | +0.0593 | 35 | 97.2% | yes |
| Rudin\|exercise_3_22 | yes | 0.9592 | 0.6006 | +0.0593 | 35 | 97.2% | yes |
| Rudin\|exercise_2_29 | yes | 0.8989 | 0.6006 | +0.0593 | 35 | 97.2% | yes |
| Rudin\|exercise_1_2 | yes | 0.9152 | 0.6006 | +0.0593 | 35 | 97.2% | yes |
| Rudin\|exercise_1_13 | yes | 0.7527 | 0.6006 | +0.0593 | 35 | 97.2% | yes |
| Putnam\|exercise_2010_a4 | yes | 0.9493 | 0.6006 | +0.0593 | 35 | 97.2% | yes |
| Putnam\|exercise_2000_a2 | yes | 0.9142 | 0.6006 | +0.0593 | 35 | 97.2% | yes |
| Munkres\|exercise_38_6 | yes | 0.8400 | 0.6006 | +0.0593 | 35 | 97.2% | yes |
| Munkres\|exercise_33_7 | yes | 0.9650 | 0.6006 | +0.0593 | 35 | 97.2% | yes |
| Munkres\|exercise_31_3 | yes | 0.9454 | 0.6006 | +0.0593 | 35 | 97.2% | yes |
| Munkres\|exercise_30_10 | yes | 0.9320 | 0.6006 | +0.0593 | 35 | 97.2% | yes |
| Munkres\|exercise_28_6 | yes | 0.7637 | 0.6006 | +0.0593 | 35 | 97.2% | yes |
| Munkres\|exercise_27_4 | yes | 0.9669 | 0.6006 | +0.0593 | 35 | 97.2% | yes |
| Munkres\|exercise_22_2a | yes | 0.7692 | 0.6006 | +0.0593 | 35 | 97.2% | yes |
