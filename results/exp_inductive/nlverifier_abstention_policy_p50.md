# NLVerifier Abstention Policy

- Confidence key: `chosen_probability`
- Threshold: `0.5413`
- Threshold source: `target_recommendation`
- Target accuracy: 50.0%
- Wilson-LCB certified target: yes
- Problems: 55
- Accepted: 36 (65.5%)
- Abstained: 19 (34.5%)
- Full-coverage accuracy: 49.1% (27/55)
- Accepted accuracy: 66.7% [50.3%, 79.8%] (24/36)
- Accepted oracle ceiling: 69.4% (25/36)

## Outcome Counts

| bucket | count | selected correct | has any correct | missed available correct | no correct candidate | accuracy | oracle ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|
| accepted | 36 | 24 | 25 | 1 | 11 | 66.7% | 69.4% |
| abstained | 19 | 3 | 4 | 1 | 15 | 15.8% | 21.1% |

## Accepted Errors

| problem_id | confidence | has any correct | chosen |
|---|---:|---:|---:|
| Dummit-Foote\|exercise_8_3_6b | 0.9674 | no | 0 |
| Axler\|exercise_7_10 | 0.9637 | no | 0 |
| Herstein\|exercise_4_3_25 | 0.9502 | yes | 3 |
| Dummit-Foote\|exercise_2_4_16b | 0.9496 | no | 0 |
| Herstein\|exercise_5_6_14 | 0.9040 | no | 8 |
| Axler\|exercise_6_2 | 0.8838 | no | 5 |
| Dummit-Foote\|exercise_4_4_7 | 0.8397 | no | 1 |
| Putnam\|exercise_2018_b2 | 0.7879 | no | 0 |
| Herstein\|exercise_2_5_43 | 0.7273 | no | 1 |
| Dummit-Foote\|exercise_9_4_11 | 0.6303 | no | 7 |
| Pugh\|exercise_2_29 | 0.6006 | no | 15 |
| Dummit-Foote\|exercise_8_2_4 | 0.5413 | no | 4 |

## Highest-Confidence Abstentions

| problem_id | confidence | selected correct | has any correct | chosen |
|---|---:|---:|---:|---:|
| Ireland-Rosen\|exercise_4_11 | 0.5132 | no | no | 4 |
| Munkres\|exercise_13_4b1 | 0.4406 | no | no | 6 |
| Ireland-Rosen\|exercise_5_37 | 0.3953 | no | no | 3 |
| Pugh\|exercise_3_1 | 0.3806 | yes | yes | 9 |
| Axler\|exercise_7_14 | 0.3481 | no | no | 0 |
| Dummit-Foote\|exercise_4_5_28 | 0.3424 | no | no | 1 |
| Dummit-Foote\|exercise_5_4_2 | 0.2763 | no | no | 2 |
| Axler\|exercise_4_4 | 0.2587 | yes | yes | 0 |
| Munkres\|exercise_20_2 | 0.2501 | no | no | 0 |
| Rudin\|exercise_3_21 | 0.2478 | no | no | 1 |
| Axler\|exercise_1_6 | 0.1744 | no | yes | 5 |
| Herstein\|exercise_5_4_3 | 0.1281 | no | no | 2 |
| Axler\|exercise_1_7 | 0.1150 | no | no | 1 |
| Dummit-Foote\|exercise_1_6_4 | 0.1062 | yes | yes | 7 |
| Herstein\|exercise_2_8_15 | 0.0928 | no | no | 0 |
