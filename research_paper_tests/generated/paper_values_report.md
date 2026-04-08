# Research Paper Numeric Results

This file is generated from the pytest-backed research harness.

Regenerate with `.venv/bin/pytest research_paper_tests/test_paper_values_report.py -q`.

## Sentiment Mapping

| Condition | Negative | Neutral | Positive |
| :--- | ---: | ---: | ---: |
| Raw | 0.500000 | 0.500000 | 0.000000 |
| Low Activity | 0.333333 | 0.666667 | 0.000000 |
| High Activity | 0.500000 | 0.500000 | 0.000000 |

- Neutral gain at low activity vs raw: `0.166667`
- Negative drop at low activity vs raw: `0.166667`
- Neutral gain at low activity vs high activity: `0.166667`

## Semantic Alignment

- Positive world Wasserstein match: `0.293700`
- Positive world mismatch against negative baseline: `1.306300`
- Negative world Wasserstein match: `0.435600`
- Negative world mismatch against positive baseline: `1.164400`
- Negative-minus-positive negative sentiment share: `0.365898`

## Accuracy Metrics

- Matching Wasserstein distance: `0.546500`
- Mismatched Wasserstein distance: `1.153500`
- Wasserstein gap: `0.607000`

## Response Boundaries

| Magnitude | Mean Engagement | Acting Ratio | Sentiment Valence |
| :--- | ---: | ---: | ---: |
| 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.150000 | 0.106356 | 0.777344 | -0.110000 |
| 0.300000 | 0.231087 | 0.847656 | -0.204000 |
| 0.450000 | 0.327931 | 0.882812 | -0.280000 |
| 0.600000 | 0.401527 | 0.898438 | -0.339000 |
| 0.750000 | 0.458063 | 0.921875 | -0.384000 |
| 0.900000 | 0.502383 | 0.925781 | -0.420000 |

### Low-Salience Worlds

| World | Mean Engagement | Acting Ratio | Sentiment Valence |
| :--- | ---: | ---: | ---: |
| Zero | 0.000000 | 0.000000 | 0.000000 |
| Faint Threat | 0.007493 | 0.000000 | -0.019000 |
| Mixed Weak | 0.011567 | 0.000000 | -0.002000 |
| Salient Threat | 0.384224 | 0.890625 | -0.320000 |

## Emotion Directionality

| World | Dominant Emotion | Acting Ratio | Sentiment Valence |
| :--- | :--- | ---: | ---: |
| Prosperity | Joy | 0.886719 | 0.711000 |
| Threat | Fear | 0.921875 | -0.369000 |
| Injustice | Anger | 0.898438 | -0.361000 |

## Bridge Diffusion

- Acting ratio without bridge: `0.600000`
- Acting ratio with bridge: `1.000000`
- Acting ratio gain: `0.400000`
- Community-B local arousal gain: `0.068000`

## Inequality And Topology

- Baseline wealth Gini: `0.199135`
- Evolved wealth Gini: `0.323614`
- Wealth Gini delta: `0.124479`
- Backbone clustering: `0.065414`
- Clustering with triadic closure: `0.615779`
- Clustering gain: `0.550366`
- Low-homophily Louvain modularity: `0.099969`
- High-homophily Louvain modularity: `0.285240`
- Modularity gain: `0.185271`

## Memory, Amplification, And Virality

- Memory final norm gain from rehearsal: `5.592401`
- Algorithmic amplification engagement gain: `0.044439`
- Algorithmic amplification max world shift: `0.100000`
- Configured viral cap: `11.000000`
- Peak viral slope: `39.125000`

