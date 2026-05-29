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

- Positive world Wasserstein match: `0.283800`
- Positive world mismatch against negative baseline: `1.316200`
- Negative world Wasserstein match: `0.433900`
- Negative world mismatch against positive baseline: `1.166100`
- Negative-minus-positive negative sentiment share: `0.373362`

## Accuracy Metrics

- Matching Wasserstein distance: `0.544500`
- Mismatched Wasserstein distance: `1.155500`
- Wasserstein gap: `0.611000`

## Response Boundaries

| Magnitude | Mean Engagement | Acting Ratio | Sentiment Valence |
| :--- | ---: | ---: | ---: |
| 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.150000 | 0.106133 | 0.773438 | -0.118000 |
| 0.300000 | 0.230850 | 0.847656 | -0.227000 |
| 0.450000 | 0.327785 | 0.886719 | -0.321000 |
| 0.600000 | 0.401611 | 1.000000 | -0.393000 |
| 0.750000 | 0.458504 | 1.000000 | -0.447000 |
| 0.900000 | 0.503015 | 1.000000 | -0.488000 |

### Low-Salience Worlds

| World | Mean Engagement | Acting Ratio | Sentiment Valence |
| :--- | ---: | ---: | ---: |
| Zero | 0.000000 | 0.000000 | 0.000000 |
| Faint Threat | 0.007371 | 0.000000 | -0.020000 |
| Mixed Weak | 0.011636 | 0.000000 | -0.002000 |
| Salient Threat | 0.383651 | 1.000000 | -0.375000 |

## Emotion Directionality

| World | Dominant Emotion | Acting Ratio | Sentiment Valence |
| :--- | :--- | ---: | ---: |
| Prosperity | Joy | 1.000000 | 0.831000 |
| Threat | Fear | 1.000000 | -0.439000 |
| Injustice | Anger | 1.000000 | -0.417000 |

## Bridge Diffusion

- Acting ratio without bridge: `0.600000`
- Acting ratio with bridge: `1.000000`
- Acting ratio gain: `0.400000`
- Community-B local arousal gain: `0.068000`

## Inequality And Topology

- Baseline wealth Gini: `0.199135`
- Evolved wealth Gini: `0.357948`
- Wealth Gini delta: `0.158813`
- Backbone clustering: `0.065598`
- Clustering with triadic closure: `0.810497`
- Clustering gain: `0.744899`
- Low-homophily Louvain modularity: `0.138839`
- High-homophily Louvain modularity: `0.333521`
- Modularity gain: `0.194682`

## Memory, Amplification, And Virality

- Memory final norm gain from rehearsal: `5.592401`
- Algorithmic amplification engagement gain: `0.064467`
- Algorithmic amplification max world shift: `0.000000`
- Configured viral cap: `13.000000`
- Peak viral slope: `57.950000`

