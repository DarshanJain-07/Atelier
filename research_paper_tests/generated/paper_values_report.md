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

- Positive world Wasserstein match: `0.281300`
- Positive world mismatch against negative baseline: `1.318700`
- Negative world Wasserstein match: `0.434100`
- Negative world mismatch against positive baseline: `1.165900`
- Negative-minus-positive negative sentiment share: `0.374085`

## Accuracy Metrics

- Matching Wasserstein distance: `0.544500`
- Mismatched Wasserstein distance: `1.155500`
- Wasserstein gap: `0.611000`

## Response Boundaries

| Magnitude | Mean Engagement | Acting Ratio | Sentiment Valence |
| :--- | ---: | ---: | ---: |
| 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.150000 | 0.116911 | 0.773438 | -0.117000 |
| 0.300000 | 0.250470 | 0.847656 | -0.226000 |
| 0.450000 | 0.353340 | 0.886719 | -0.321000 |
| 0.600000 | 0.431432 | 0.984085 | -0.394000 |
| 0.750000 | 0.491371 | 0.986700 | -0.448000 |
| 0.900000 | 0.538234 | 0.988591 | -0.489000 |

### Low-Salience Worlds

| World | Mean Engagement | Acting Ratio | Sentiment Valence |
| :--- | ---: | ---: | ---: |
| Zero | 0.000000 | 0.000000 | 0.000000 |
| Faint Threat | 0.008293 | 0.000000 | -0.020000 |
| Mixed Weak | 0.013453 | 0.000000 | -0.002000 |
| Salient Threat | 0.412267 | 0.983960 | -0.376000 |

## Emotion Directionality

| World | Dominant Emotion | Acting Ratio | Sentiment Valence |
| :--- | :--- | ---: | ---: |
| Prosperity | Joy | 0.992171 | 0.837000 |
| Threat | Fear | 0.989848 | -0.439000 |
| Injustice | Anger | 0.986239 | -0.417000 |

## Bridge Diffusion

- Acting ratio without bridge: `0.600000`
- Acting ratio with bridge: `0.956989`
- Acting ratio gain: `0.356989`
- Community-B local arousal gain: `0.068000`

## Inequality And Topology

- Baseline wealth Gini: `0.199135`
- Evolved wealth Gini: `0.357948`
- Wealth Gini delta: `0.158813`
- Backbone clustering: `0.178197`
- Clustering with triadic closure: `0.790624`
- Clustering gain: `0.612427`
- Low-homophily Louvain modularity: `0.134352`
- High-homophily Louvain modularity: `0.279722`
- Modularity gain: `0.145371`

## Memory, Amplification, And Virality

- Memory final norm gain from rehearsal: `4.461011`
- Algorithmic amplification engagement gain: `0.025087`
- Algorithmic amplification max world shift: `0.000000`
- Configured viral cap: `13.000000`
- Peak viral slope: `57.950000`

