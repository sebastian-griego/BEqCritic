# NLVerifier Threshold Recommendations

- Problems: 55
- Any-correct ceiling: 52.7% (29/55)
- Full-coverage selected accuracy: 49.1% (27/55)
- Minimum accepted examples: `5`

## Recommendations

| confidence key | target accuracy | certified? | accepted | coverage | accuracy | 95% LCB | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| `chosen_probability` | 50.0% | yes | 36 | 65.5% | 66.7% | 50.3% | 0.5413 |
| `chosen_probability` | 60.0% | no | 32 | 58.2% | 75.0% | 57.9% | 0.7527 |
| `chosen_probability` | 70.0% | no | 32 | 58.2% | 75.0% | 57.9% | 0.7527 |
| `probability_margin` | 50.0% | no | 7 | 12.7% | 85.7% | 48.7% | 0.4036 |
| `probability_margin` | 60.0% | no | 7 | 12.7% | 85.7% | 48.7% | 0.4036 |
| `probability_margin` | 70.0% | no | 7 | 12.7% | 85.7% | 48.7% | 0.4036 |
| `score_margin` | 50.0% | no | 8 | 14.5% | 75.0% | 40.9% | 4.1589 |
| `score_margin` | 60.0% | no | 8 | 14.5% | 75.0% | 40.9% | 4.1589 |
| `score_margin` | 70.0% | no | 8 | 14.5% | 75.0% | 40.9% | 4.1589 |

## Best Lower-Bound Prefixes

| confidence key | accepted | coverage | accuracy | 95% LCB | threshold |
|---|---:|---:|---:|---:|---:|
| `chosen_probability` | 32 | 58.2% | 75.0% | 57.9% | 0.7527 |
| `probability_margin` | 7 | 12.7% | 85.7% | 48.7% | 0.4036 |
| `score_margin` | 8 | 14.5% | 75.0% | 40.9% | 4.1589 |
