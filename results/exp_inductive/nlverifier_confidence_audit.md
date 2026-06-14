# NLVerifier Confidence Signal Audit

- Problems: 55
- Any-correct ceiling: 52.7% (29/55)
- Full-coverage selected accuracy: 49.1% (27/55)
- Best mean prefix risk: `chosen_probability`
- Best oracle-normalized accuracy area: `chosen_probability`

## Signal Summary

| confidence key | mean prefix risk | mean prefix accuracy | lift over full | average precision | oracle-normalized area | best prefix accuracy | best prefix coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `chosen_probability` | 35.8% | 64.2% | +15.2 pp | 69.8% | 71.8% | 78.9% | 34.5% |
| `probability_margin` | 48.4% | 51.6% | +2.5 pp | 55.6% | 53.2% | 100.0% | 3.6% |
| `score_margin` | 47.6% | 52.4% | +3.3 pp | 56.0% | 54.4% | 100.0% | 3.6% |

## Coverage Comparison

| target coverage | `chosen_probability` accuracy | `probability_margin` accuracy | `score_margin` accuracy | best key |
|---:|---:|---:|---:|---|
| 25.0% | 71.4% | 50.0% | 50.0% | `chosen_probability` |
| 50.0% | 71.4% | 42.9% | 39.3% | `chosen_probability` |
| 75.0% | 59.5% | 47.6% | 45.2% | `chosen_probability` |
| 100.0% | 49.1% | 49.1% | 49.1% | `chosen_probability` |

## Coverage Details

### chosen_probability

| accepted | coverage | accuracy | risk | threshold |
|---:|---:|---:|---:|---:|
| 14 | 25.5% | 71.4% | 28.6% | 0.9327 |
| 28 | 50.9% | 71.4% | 28.6% | 0.7783 |
| 42 | 76.4% | 59.5% | 40.5% | 0.3424 |
| 55 | 100.0% | 49.1% | 50.9% | 0.0701 |

### probability_margin

| accepted | coverage | accuracy | risk | threshold |
|---:|---:|---:|---:|---:|
| 14 | 25.5% | 50.0% | 50.0% | 0.2122 |
| 28 | 50.9% | 42.9% | 57.1% | 0.0456 |
| 42 | 76.4% | 47.6% | 52.4% | 0.0073 |
| 55 | 100.0% | 49.1% | 50.9% | 0.0000 |

### score_margin

| accepted | coverage | accuracy | risk | threshold |
|---:|---:|---:|---:|---:|
| 14 | 25.5% | 50.0% | 50.0% | 2.7276 |
| 28 | 50.9% | 39.3% | 60.7% | 1.0496 |
| 42 | 76.4% | 45.2% | 54.8% | 0.2083 |
| 55 | 100.0% | 49.1% | 50.9% | 0.0000 |
