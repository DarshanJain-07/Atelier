I used `ls -la` first, then repository file inventory and local code/content inspection. There is no registered MCP/codebase-research resource in this session, so I used shell/AST/text/tensor/parquet inspection instead. I treated `.git/` and `.venv/` as internals/dependencies, not project implementation, and I am not reproducing `.env` contents because it may contain credentials.

Below is the code-grounded “write it from scratch” version I would produce.

# ATELIER: A Full Technical Research Monograph

## Title

**ATELIER: A Hybrid Neuro-Symbolic Multi-Agent Framework for Forecasting Public Sentiment Dynamics, Emotional Contagion, Polarization, Structural Inequality, and Endogenous Social Backlash**

## Abstract

ATELIER is a hybrid neuro-symbolic multi-agent simulation system for studying how a natural-language event becomes structured social perception, how that perception is distorted by psychology and social topology, and how individual emotional responses aggregate into measurable social outcomes. The system decouples semantic understanding from population-scale reaction. A calibrated LLM converts a user-provided event into a 12-dimensional world-state tensor, urgency score, personal-relevance flag, detected bias labels, and reasoning string. This compact symbolic representation is then processed by deterministic PyTorch modules that simulate thousands of heterogeneous agents with continuous exposures, Big Five personality traits, cognitive affinities, influence, wealth, class membership, network topology, memory, and generational evolution.

The project implements society generation, wealth and influence modeling, topology construction, triadic closure, personality socialization, stochastic signal distortion, social-consensus perception, relative-deprivation attention, personality-conditioned query formation, skepticism-based truth refinement, personal-event localization, selective exposure, threat-sensitive key processing, cross-dimensional interactions, engagement gating, stress-induced cognitive tunneling, memory desensitization, trigger stacking, emotion projection, virality, stewing dynamics, elite-population divergence, polarization, Granovetter-style collective action thresholds, endogenous macro-event generation, validation against a RoBERTa sentiment baseline, explanation generation, a FastAPI orchestration layer, a static browser UI, a docs browser, raw generated society artifacts, and a large pytest-based research validation suite.

This version of the paper would be intentionally exhaustive. It would document not only the intended theory but also the actual implementation: the tensor shapes, matrices, configuration fields, formulas, endpoints, UI behavior, generated figures, raw parquet/tensor data, regression tests, and implementation caveats.

## Local Source Basis

Primary files inspected include [README.md](/home/darshan/Desktop/djsce/BE_Project/README.md), [schema.py](/home/darshan/Desktop/djsce/BE_Project/schema.py), [input_layer.py](/home/darshan/Desktop/djsce/BE_Project/input_layer.py), [generate_society.py](/home/darshan/Desktop/djsce/BE_Project/generate_society.py), [society_evolution.py](/home/darshan/Desktop/djsce/BE_Project/society_evolution.py), [attention_context.py](/home/darshan/Desktop/djsce/BE_Project/attention_context.py), [cognitive_engine.py](/home/darshan/Desktop/djsce/BE_Project/cognitive_engine.py), [physics_engine.py](/home/darshan/Desktop/djsce/BE_Project/physics_engine.py), [validation.py](/home/darshan/Desktop/djsce/BE_Project/validation.py), [explainability.py](/home/darshan/Desktop/djsce/BE_Project/explainability.py), [main.py](/home/darshan/Desktop/djsce/BE_Project/main.py), [docs_router.py](/home/darshan/Desktop/djsce/BE_Project/docs_router.py), [tracer.py](/home/darshan/Desktop/djsce/BE_Project/tracer.py), [run_all_tests.sh](/home/darshan/Desktop/djsce/BE_Project/run_all_tests.sh), [research_paper.tex](/home/darshan/Desktop/djsce/BE_Project/research_paper.tex), [research_paper_modularity_results.md](/home/darshan/Desktop/djsce/BE_Project/research_paper_modularity_results.md), the docs in [docs](/home/darshan/Desktop/djsce/BE_Project/docs), the frontend in [frontend](/home/darshan/Desktop/djsce/BE_Project/frontend), the research suite in [research_paper_tests](/home/darshan/Desktop/djsce/BE_Project/research_paper_tests), and generated society data in [society_data](/home/darshan/Desktop/djsce/BE_Project/society_data).

## 1. Introduction

Modern public sentiment is not a scalar. The same policy, disaster, corporate announcement, or geopolitical event can generate joy in one structural group, fear in another, anger in a marginalized class, indifference among insulated elites, and viral outrage among highly engaged agents. Traditional sentiment systems compress these reactions into a single post-hoc label, usually positive, neutral, or negative. That loses heterogeneity, network structure, memory, class, influence, exposure, topology, and endogenous feedback.

ATELIER models this problem as a layered simulation. The natural-language event is parsed once into a structured symbolic tensor. The social world then responds through deterministic, inspectable tensor mechanics. This gives the system three important properties:

| Property | Implementation |
| --- | --- |
| Semantic grounding | LLM parses raw event text into a calibrated 12-dimensional world tensor. |
| Population scale | PyTorch tensors represent thousands of agents without querying an LLM per agent. |
| Explainability | Every downstream mechanism is explicit: personality, topology, memory, attention, virality, class, and action thresholds. |

The system is not merely a sentiment classifier. It is a simulation framework for ex-ante stress testing: “If this event is introduced into a stratified society with these psychological traits, wealth distributions, network topology, and platform amplification dynamics, what collective emotional and behavioral state emerges?”

## 2. System Overview

The runtime pipeline is:

| Stage | Responsibility |
| --- | --- |
| Perception | Convert text into world tensor, urgency, personal flag, bias labels, reasoning. |
| Society generation | Create agents with exposures, personalities, affinities, influence, wealth, class, topology. |
| Society evolution | Optionally run multi-generation wealth, influence, ideology, mobility, and personality drift. |
| Cognitive processing | Convert objective event into each agent’s perceived context, attention, engagement, and memory-modulated reaction. |
| Emotion projection | Map 12-dimensional context into 8 Plutchik-style emotion probabilities. |
| Social physics | Aggregate emotions into objective center, viral center, elite center, virality, polarization, action readiness, and endogenous events. |
| Validation | Compare model output to a Hugging Face RoBERTa sentiment baseline with Wasserstein and KL divergence. |
| Explainability | Generate human-readable summaries for cognitive drivers, virality, demographic archetypes, social structure, and endogenous events. |
| Orchestration | Serve FastAPI endpoints, run tasks concurrently, cache societies, slice classes, mount frontend and docs. |
| UI | Render simulation controls, canvas visualization, run filmstrip, history, explanations, and agent dossiers. |

## 3. Core Schema

ATELIER’s world representation is a 12-dimensional vector:

| Index | Dimension | Meaning |
| ---: | --- | --- |
| 0 | Wealth | Economic resources, prosperity, loss, capital. |
| 1 | Physical_Safety | Bodily safety, harm, danger. |
| 2 | Stability | Order, predictability, institutional/social continuity. |
| 3 | Reputation | Prestige, status, social standing. |
| 4 | Fairness | Justice, exploitation, procedural and distributive fairness. |
| 5 | In_Group | Cohesion, tribal loyalty, group belonging. |
| 6 | Innovation | Novelty, progress, technological or social change. |
| 7 | Freedom | Autonomy, coercion, liberty. |
| 8 | Sanctity | Purity, taboo, moral contamination. |
| 9 | Care | Empathy, protection, harm reduction. |
| 10 | Short_Term | Immediate/transient impact. |
| 11 | Long_Term | Structural/generational impact. |

The emotional representation is 8-dimensional:

| Index | Emotion |
| ---: | --- |
| 0 | Joy |
| 1 | Trust |
| 2 | Fear |
| 3 | Surprise |
| 4 | Sadness |
| 5 | Disgust |
| 6 | Anger |
| 7 | Anticipation |

The valence weights are:

```text
Joy=+1.0
Trust=+0.5
Fear=-0.5
Surprise=0.0
Sadness=-0.5
Disgust=-0.5
Anger=-0.5
Anticipation=+0.5
```

Valence is computed as:

$$
Valence(E) = E \cdot [1.0, 0.5, -0.5, 0.0, -0.5, -0.5, -0.5, 0.5]
$$

Emotion-to-sentiment mapping converts 8 emotions into 3 sentiment buckets: negative, neutral, positive. Behavior-aware sentiment additionally increases neutrality when the acting ratio is below a configured threshold.

## 4. Psychological Projection Matrix

The core mapping from 12 world dimensions to 8 emotion logits is implemented as a fixed matrix:

| Dimension | Joy | Trust | Fear | Surprise | Sadness | Disgust | Anger | Anticipation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wealth | 0.6 | 0.2 | 0.0 | 0.0 | -0.6 | 0.0 | -0.2 | 0.1 |
| Physical_Safety | 0.1 | 0.3 | -0.8 | 0.2 | -0.2 | 0.0 | -0.2 | 0.0 |
| Stability | 0.1 | 0.6 | -0.3 | 0.0 | -0.2 | 0.0 | -0.3 | 0.0 |
| Reputation | 0.5 | 0.3 | -0.1 | 0.0 | -0.3 | -0.1 | -0.2 | 0.0 |
| Fairness | 0.1 | 0.4 | 0.0 | 0.0 | -0.2 | -0.4 | -0.8 | 0.0 |
| In_Group | 0.1 | 0.8 | -0.2 | 0.0 | -0.2 | -0.3 | 0.0 | 0.0 |
| Innovation | 0.4 | 0.1 | -0.1 | 0.6 | 0.0 | 0.0 | 0.0 | 0.5 |
| Freedom | 0.6 | 0.1 | -0.2 | 0.1 | -0.2 | -0.1 | -0.5 | 0.0 |
| Sanctity | 0.0 | 0.4 | 0.0 | 0.0 | 0.0 | -0.8 | -0.4 | 0.0 |
| Care | 0.2 | 0.5 | -0.1 | 0.0 | -0.5 | -0.1 | -0.1 | 0.0 |
| Short_Term | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 | 0.0 | 0.0 | 0.6 |
| Long_Term | 0.0 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.8 |

Projection is:

$$
EmotionLogits_i = Context_i \cdot P
$$

$$
Emotion_i = Softmax\left(\frac{EmotionLogits_i}{temperature}\right)
$$

## 5. Big Five Personality Representation

Each agent has an OCEAN personality vector:

```text
[Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism]
```

The target correlation matrix is:

| Trait | O | C | E | A | N |
| --- | ---: | ---: | ---: | ---: | ---: |
| O | 1.00 | 0.01 | 0.35 | 0.01 | -0.01 |
| C | 0.01 | 1.00 | 0.01 | 0.40 | -0.40 |
| E | 0.35 | 0.01 | 1.00 | 0.01 | -0.01 |
| A | 0.01 | 0.40 | 0.01 | 1.00 | -0.40 |
| N | -0.01 | -0.40 | -0.01 | -0.40 | 1.00 |

The society generator samples raw traits, applies Cholesky decomposition of this matrix, then passes logits through sigmoid to produce bounded traits.

## 6. Personality Query Matrix

Personality modulates attention through a fixed 5-by-12 matrix:

| Trait | Wealth | Safety | Stability | Reputation | Fairness | In_Group | Innovation | Freedom | Sanctity | Care | Short_Term | Long_Term |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Openness | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.8 | 0.5 | 0.0 | 0.0 | 0.0 | 0.6 |
| Conscientiousness | 0.6 | 0.0 | 0.0 | 0.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Extraversion | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | 0.0 | 0.0 | 0.0 | 0.4 | 0.0 | 0.0 |
| Agreeableness | 0.0 | 0.0 | 0.0 | 0.0 | 0.8 | 0.0 | 0.0 | 0.0 | 0.5 | 0.6 | 0.0 | 0.0 |
| Neuroticism | 0.0 | 1.2 | 0.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

This means high openness increases attention to innovation, freedom, and long-term effects; high conscientiousness prioritizes wealth and reputation; high agreeableness prioritizes fairness, sanctity, and care; high neuroticism prioritizes safety and stability.

## 7. Cross-Dimension Interactions

The implemented cross-dimension interaction matrix is sparse:

| Source | Target | Weight |
| --- | --- | ---: |
| Physical_Safety | Stability | 0.4 |
| Physical_Safety | Care | 0.3 |
| Wealth | Reputation | 0.3 |
| Wealth | Freedom | 0.2 |
| Innovation | Short_Term | 0.2 |
| Innovation | Long_Term | 0.3 |

The attention pipeline computes:

$$
Q_{cross} = (Q \cdot strength) \cdot C
$$

$$
Q' = tanh(Q + Q_{cross})
$$

## 8. Runtime Configuration

The central `SimConfig` contains infrastructure settings, feature toggles, cognitive parameters, memory parameters, topology parameters, evolution parameters, stewing parameters, physics parameters, cascade parameters, and validation constraints.

Important defaults include:

| Field | Default | Role |
| --- | ---: | --- |
| seed | 42 | Deterministic generation seed. |
| num_agents | 10000 | Default population size. |
| output_dir | society_data | Default artifact directory. |
| use_signal_distortion | True | Enables telephone-game distortion. |
| distortion_max_noise | 0.4 | Noise amplitude cap. |
| distortion_beta_a | 2.0 | Beta distribution alpha. |
| distortion_beta_b | 5.0 | Beta distribution beta. |
| distortion_neurotic_gain | 0.6 | Neuroticism distortion amplifier. |
| use_engagement_gate | True | Enables relevance-to-engagement gate. |
| engagement_threshold | 0.15 | Energy threshold for engagement. |
| engagement_gain | 10.0 | Sigmoid sharpness. |
| use_selective_exposure | True | Enables filter-bubble suppression. |
| use_time_pressure | True | Enables urgency-driven stress bias. |
| use_social_conformity | False | Optional conformity toward population mean. |
| use_agent_memory | False | Enables memory tensor. |
| use_algorithmic_amplification | False | Enables two-pass feed mutation. |
| use_power_law_influence | False | Enables Pareto influence multiplier. |
| use_network_topology | True | Enables sparse graph generation. |
| homophily_strength | 6.0 | Similarity exponent for topology. |
| triadic_closure_prob | 0.2 | Friend-of-friend closure probability. |
| enable_evolution | True | Runs generational evolution before simulation. |
| evolution_generations | 10 | Number of evolution steps. |
| stewing_ticks | 5 | Social-physics iteration count. |
| outrage_gain | 8.0 | Viral sigmoid gain. |
| max_viral_multiplier | 10.0 | Viral amplification cap. |
| elite_percentile | 0.95 | Top 5 percent define elite center. |
| dominant_emotion_threshold | 0.155 | Minimum emotion confidence for non-neutral dominance. |
| polarization_threshold | 0.5 | Endogenous action threshold component. |
| action_threshold | 0.15 | Acting-ratio trigger threshold. |
| use_granovetter_thresholds | True | Enables local threshold cascades. |

Implementation note: `cascade_knn_k` is defined but not materially active in the current backend runtime.

## 9. Perception Layer

The perception layer uses `gemini-3-flash-preview` through Google’s Generative Language API. It sends a structured prompt instructing the model to act as a predictive world model rather than ordinary sentiment analysis.

The response schema requires:

```text
Reasoning: string
Detected_Biases: list[string]
Urgency: float
Is_Personal: bool
Wealth: float
Physical_Safety: float
Stability: float
Reputation: float
Fairness: float
In_Group: float
Innovation: float
Freedom: float
Sanctity: float
Care: float
Short_Term: float
Long_Term: float
```

The LLM magnitude rubric is:

| Score | Meaning |
| ---: | --- |
| 0.0 | Neutral or no relevance. |
| +/-0.1 to +/-0.2 | Routine or minor event. |
| +/-0.3 to +/-0.5 | Significant event. |
| +/-0.6 to +/-0.8 | Crisis or boom. |
| +/-0.9 to +/-1.0 | Civilization-altering event. |

The perception layer also detects framing such as corporate spin, political framing, imperial or condescending language, fearmongering, and downplaying.

The implementation includes an in-memory LLM cache keyed by the raw input string, three retries with exponential delay, a 10-second request timeout, strict Pydantic validation with `extra="forbid"`, and final clamping of world tensor values to `[-1, 1]` and urgency to `[0, 1]`.

## 10. Society Generation

The society generator creates:

| Artifact | Shape / Type | Meaning |
| --- | --- | --- |
| exposures | `(N, 12)` dense tensor | Agent worldview/exposure vector. |
| personalities | `(N, 5)` dense tensor | OCEAN traits in `[0,1]`. |
| affinities | `(N, 12)` dense tensor | Cognitive bandwidth and dimension affinity. |
| metadata | DataFrame/parquet | Agent id, class, region, influence, raw wealth, bandwidth, structural metrics. |
| adjacency | `(N, N)` sparse tensor | Row-normalized topology, if enabled. |

The initial trait tensor has:

$$
total\_dims = 12 + 5 + 12 = 29
$$

It samples:

$$
traits \sim Normal(0, initial\_trait\_std\_dev)
$$

The first 12 dimensions initialize exposures. Wealth is temporarily set to zero and filled later from raw structural wealth.

The next 5 dimensions initialize personality logits. They are scaled and correlated:

$$
raw\_personalities = \frac{traits_{personality}}{initial\_trait\_std\_dev} \cdot 1.5
$$

$$
raw\_personalities' = raw\_personalities \cdot Cholesky(PERSONALITY\_CORRELATIONS + \epsilon I)^T
$$

$$
personalities = sigmoid(raw\_personalities')
$$

The final 12 dimensions initialize raw affinities. Affinity is converted into positive per-dimension bandwidth:

$$
positive\_affinity = clamp(|raw\_affinity|, min=affinity\_min\_strength)
$$

If mean normalization is enabled:

$$
normalized\_affinity_{i,d} =
\frac{positive\_affinity_{i,d}}{mean_d(positive\_affinity_i)}
$$

Cognitive bandwidth is sampled as:

$$
bandwidth_i = clamp(Normal(0.55, 0.2), 0.1, 1.0)
$$

Final affinity is:

$$
affinity_{i,d} = normalized\_affinity_{i,d} \cdot bandwidth_i
$$

## 11. Mutation Logic

If mutation temperature is greater than zero, the generator randomly mutates exposures and personality logits.

Mutation probability is:

$$
P(mutant) = temperature
$$

The number of changes per mutant is:

$$
num\_changes = ceil(3 \cdot temperature)
$$

Exposure mutations sample random world dimensions and replace them with clipped normal values:

$$
value \sim clamp(Normal(0, 0.4), -1, 1)
$$

Personality mutations add normal noise:

$$
logit' = logit + Normal(0, 0.5)
$$

Radical outliers occur with:

$$
P(radical) = 0.05 \cdot temperature
$$

Radical personality logits are overwritten by:

$$
Normal(0, 2.5)
$$

## 12. Influence Generation

Baseline influence is sampled as:

$$
Influence_i \sim LogNormal(1.0, 0.5 + mutation\_temperature)
$$

If power-law influence is enabled:

$$
Influence_i' = Influence_i \cdot ((Pareto(1.16) + 1) \cdot 2)
$$

This creates the heavier-tailed influence distributions tested in the research suite.

## 13. Structural Wealth Generation

Raw wealth is generated before topology and class assignment.

Trait-derived merit is:

$$
merit_i =
0.45C_i + 0.20(1-N_i) + 0.20E_i + 0.15O_i
$$

Influence percentile is computed through stable percentile ranks. Seed potential is:

$$
SeedPotential_i =
800 + 2800 \cdot merit_i + 1800 \cdot Influence_i^{0.70}
$$

Realization noise is:

$$
RealizationNoise_i \sim LogNormal(0, 0.18 + 0.22 \cdot temperature)
$$

Initial wealth is:

$$
Wealth_i = SeedPotential_i \cdot RealizationNoise_i
$$

A Pareto legacy injection is then added:

$$
\alpha = max(1.2, 2.3 - 0.5 \cdot temperature)
$$

$$
elite\_gate_i = clamp(0.55 \cdot InfluencePercentile_i + 0.45 \cdot merit_i, 0, 1)
$$

$$
Legacy_i = (Pareto(\alpha) + 1) \cdot 4500 \cdot elite\_gate_i \cdot (0.4 + 0.6 \cdot temperature)
$$

$$
Wealth_i' = max(Wealth_i + Legacy_i, 500)
$$

Raw wealth is stored in metadata. Exposure-space wealth is log-normalized:

$$
logwealth_i = log(1 + max(RawWealth_i, 0))
$$

$$
WealthExposure_i =
2 \cdot \frac{logwealth_i - min(logwealth)}{max(logwealth)-min(logwealth)} - 1
$$

## 14. Topology Construction

If topology is enabled, ATELIER builds a sparse row-normalized adjacency matrix.

Topology uses exposure and personality features, but wealth exposure is zeroed for the feature vector so that topology is not mechanically dominated by the wealth dimension:

$$
features_i = concat(exposures_i^{wealth=0}, personalities_i)
$$

$$
featuresNorm_i = \frac{features_i}{||features_i|| + \epsilon}
$$

Bridge agents are selected from top wealth and influence candidates, filtered by openness >= 0.55, or the most open among candidates if none qualify.

Each agent receives a dynamic connection target:

$$
k_i =
round(base\_connections \cdot (0.45 + 1.15O_i + 0.35EliteStrength_i + 0.25Bridge_i))
$$

where:

$$
EliteStrength_i = max(WealthPercentile_i, InfluencePercentile_i)
$$

Candidate edge probability combines trait similarity, influence homophily, wealth homophily, elite cross-connections, openness, bridge effects, and target influence bias.

Trait similarity:

$$
sim_{ij} = featuresNorm_i \cdot featuresNorm_j
$$

$$
traitSimilarity_{ij} = clamp(sim_{ij}, 0, \infty)^{homophily\_strength}
$$

Influence and wealth homophily:

$$
InfluenceHomophily_{ij} = exp(-4|InfluencePercentile_i - InfluencePercentile_j|)
$$

$$
WealthHomophily_{ij} = exp(-4|WealthPercentile_i - WealthPercentile_j|)
$$

Elite terms:

$$
InfluenceElite_{ij} = \sqrt{InfluencePercentile_i \cdot InfluencePercentile_j}
$$

$$
WealthElite_{ij} = \sqrt{WealthPercentile_i \cdot WealthPercentile_j}
$$

$$
CrossElite_{ij} = \frac{1}{2}
(\sqrt{Wealth_i \cdot Influence_j} + \sqrt{Influence_i \cdot Wealth_j})
$$

The implementation combines these weighted terms, samples without replacement through `torch.multinomial`, removes self-links, applies triadic closure, and normalizes each row so each row sum is approximately 1.

## 15. Triadic Closure

Triadic closure finds paths of length two in the sparse graph:

$$
A_2 = A_{binary} \cdot A_{binary}
$$

Candidate closure edges are sampled with `triadic_closure_prob`, exclude self-loops, and optionally require feature similarity above `triadic_closure_homophily_threshold`.

If features are available:

$$
newWeight_{ij} = featuresNorm_i \cdot featuresNorm_j
$$

Edges are merged into the sparse adjacency. The process repeats for `triadic_closure_iterations` and stops if edge count exceeds:

$$
N \cdot max\_connections
$$

## 16. Structural Class Assignment

After topology, class is assigned from structural score.

If topology is present, local wealth and influence are neighbor averages:

$$
LocalWealth = A \cdot WealthPercentile
$$

$$
LocalInfluence = A \cdot InfluencePercentile
$$

Degree is out-degree plus in-degree. Degree percentile is computed.

The structural score is:

$$
Score_i =
0.24W_i + 0.16I_i + 0.22D_i + 0.20LocalW_i + 0.14LocalI_i + 0.04O_i
$$

Class thresholds are:

| Class | Condition |
| --- | --- |
| Elite | score >= 0.86, or wealth >= 0.92 and degree >= 0.65 |
| Upper Middle | score >= 0.67 and not elite |
| Middle Class | score >= 0.44 and not higher |
| Working Class | score >= 0.22 and not higher |
| Underclass | everything else |

In the current generated `society_data`, there are no `Underclass` rows. Counts are:

| Class | Count |
| --- | ---: |
| Middle Class | 4510 |
| Upper Middle | 2717 |
| Working Class | 1840 |
| Elite | 933 |

## 17. Personality Socialization

If topology exists and `personality_socialization_gain > 0`, agents partially drift toward their local network personality mean:

$$
Personality_i' =
(1-gain) \cdot Personality_i + gain \cdot (A \cdot Personality)_i
$$

Then traits are clamped to `[0.001, 0.999]`.

## 18. Society Evolution

Society evolution starts from metadata, exposures, and personalities. It derives raw wealth from `Raw_Wealth`, influence from metadata, and wealth exposure from log-normalized raw wealth.

### Idiosyncratic Drift

Each agent receives persistent multiplicative personality drift:

$$
Idio_{i,t} = exp(Normal(0, 0.015))
$$

with seed:

$$
seed + evolution\_idiosyncrasy\_seed\_offset
$$

At each generation:

$$
Personality_i' = clamp(Personality_i \odot Idio_i, 0.001, 0.999)
$$

### Inheritance and Redistribution

Inherited wealth:

$$
Inherited_i = ParentWealth_i \cdot inheritance\_fraction
$$

Total tax/loss pool:

$$
TaxPool = \sum_i(ParentWealth_i - Inherited_i)
$$

Redistribution:

$$
Redistribution = \frac{TaxPool}{N}
$$

New wealth:

$$
Wealth_i' =
clamp(Inherited_i + Redistribution + Normal(0, inheritance\_noise\_std \cdot mean(ParentWealth)), 0, \infty)
$$

### Reinvestment

Returns are:

$$
Returns_i =
base\_return\_rate +
influence\_reinvestment\_factor \cdot \frac{Influence_i}{mean(Influence)}
+ Normal(0, reinvestment\_noise\_std)
$$

Returns are clamped to `[-0.2, 0.5]`, then:

$$
Wealth_i' = Wealth_i \cdot (1 + Returns_i)
$$

### Economic Shocks

With probability `shock_frequency`, wealth is multiplied by:

$$
1 - shock\_magnitude \cdot Uniform(0.5, 1.0)
$$

Despite the docstring mentioning positive or negative shocks, the current implementation applies only a reducing multiplier.

### Social Mobility

A fraction of agents are selected:

$$
n\_movers = int(N \cdot mobility\_rate)
$$

Their influence and wealth values are randomly permuted among selected movers.

### Ideological Drift

If enabled, non-wealth exposures drift toward either the global mean or, with probability `elite_influence_drift_chance`, the elite mean.

The target variance controls a Bayesian/Kalman-style update rate:

$$
BayesianUpdateRate_d =
\frac{prior\_variance}{prior\_variance + targetVariance_d}
$$

$$
DynamicDriftRate_d =
ideological\_drift\_rate \cdot BayesianUpdateRate_d
$$

Agents move toward target:

$$
Diff_i = TargetMean - Exposure_i
$$

If ideological repulsion is enabled, alignment is cosine similarity between agent non-wealth exposure and target mean. If alignment is below `repulsion_threshold`, the agent moves away instead:

$$
Drift_i =
\begin{cases}
Diff_i \cdot DynamicDriftRate & alignment \ge threshold \\
-Diff_i \cdot repulsion\_rate & alignment < threshold
\end{cases}
$$

Noise is added, and non-wealth exposures are bounded with `tanh`.

Implementation note: there is a `reassign_classes` method in the evolution class, but the current `evolve()` loop does not call it. In the runtime path, classes are reassigned after evolution through final topology/structure finalization.

## 19. Cognitive Engine

The cognitive engine converts the world tensor into agent-specific context.

### Signal Distortion

If disabled, every agent receives the same world tensor. If enabled:

$$
\alpha_i \sim Beta(distortion\_beta\_a, distortion\_beta\_b)
$$

$$
\alpha_i' =
clamp(\alpha_i(1 + distortion\_neurotic\_gain \cdot N_i), distortion\_min\_alpha, 1)
\cdot distortion\_max\_noise
$$

The distortion is scaled by signal norm:

$$
Distorted_i =
World + \alpha_i' \cdot ||World||_2 \cdot Normal(0, I)
$$

Then clamped to `[-1.5, 1.5]`.

### Social Consensus Perception

If adjacency exists and consensus gain is positive:

$$
LocalConsensus = A \cdot DistortedWorld
$$

$$
Perceived =
(1-g)DistortedWorld + gLocalConsensus
$$

This implements socially constructed perception.

### Memory Interaction

If memory is enabled, agent memory modifies the distorted world before attention.

Fatigue alignment:

$$
Alignment = Memory_i \odot DistortedWorld_i
$$

$$
FatigueMask = clamp(Alignment, min=0)
$$

$$
FatiguePenalty = 1 - exp(-FatigueMask \cdot memory\_desensitization\_gain)
$$

Threat mask:

$$
ThreatMask = 1[DistortedWorld < 0]
$$

Total past stress:

$$
TotalStress_i = \sum_d |Memory_{i,d}|
$$

Fresh threat mask:

$$
FreshThreat = ThreatMask \odot 1[FatigueMask < 0.1]
$$

Trigger stacking:

$$
TriggerBoost =
log(1+TotalStress_i) \cdot FreshThreat \cdot memory\_trigger\_stacking\_gain
$$

Perceived world:

$$
Perceived =
DistortedWorld \cdot (1 - clamp(FatiguePenalty, max=0.95))
+
DistortedWorld \cdot TriggerBoost
$$

Then the perceived world is multiplied by agent affinities.

## 20. Attention Pipeline

The attention pipeline is modular and ordered.

### Relative Deprivation/Enhancement Layer

Exposures are normalized:

$$
NormalizedState = \frac{Exposures + 1}{2}
$$

$$
Gaps = 1 - NormalizedState
$$

$$
Assets = NormalizedState
$$

Jealousy bias:

$$
Jealousy =
Neuroticism \cdot jealousy\_factor +
(1-Agreeableness) \cdot resentment\_factor
$$

Protection bias:

$$
Protection =
Conscientiousness \cdot protection\_factor +
Extraversion \cdot status\_factor
$$

Sensitivity:

$$
RDE =
Gaps \cdot Jealousy + Assets \cdot Protection
$$

Query:

$$
Q = Exposures \cdot clamp(RDE, 0.1, 3.0)
$$

### Personality Query Layer

Trait activations are:

$$
O_{act} = sigmoid(3O - 1)
$$

$$
C_{act} = relu(C - 0.3) \cdot 1.5
$$

$$
E_{act} = sigmoid(2E)
$$

$$
A_{act} = tanh(2A)
$$

$$
N_{act} = tanh(2.5N)
$$

Then:

$$
PersonalityMod = Activations \cdot PersonalityQueryMatrix
$$

$$
Q' = tanh(Q + PersonalityMod)
$$

### Logic Consistency / Skepticism Gate

Short-term and long-term impact discrepancy:

$$
LogicGap = |World_{ShortTerm} - World_{LongTerm}|
$$

Skepticism trait:

$$
Skepticism = \frac{Openness + Conscientiousness}{2}
$$

Sensitivity:

$$
Sensitivity = Skepticism \cdot skepticism\_gain
$$

Detection probability:

$$
Detection = sigmoid(Sensitivity(LogicGap - logic\_gap\_threshold))
$$

Then short-term attention is suppressed:

$$
Q_{ShortTerm}' = Q_{ShortTerm}(1 - 0.85Detection)
$$

and long-term attention is boosted:

$$
Q_{LongTerm}' = Q_{LongTerm}(1 + 0.6Detection)
$$

### Personal Event Layer

If `is_personal` is false, the layer does nothing. If true, relevance becomes localized.

Proximity:

$$
Proximity_i = Uniform(0,1)^{10}
$$

Maximum boost:

$$
MaxBoost = exp(1.5)
$$

Agent multiplier:

$$
Multiplier_i =
0.05 + MaxBoost \cdot Proximity_i \cdot (0.5 + 0.5Agreeableness_i)
$$

Then:

$$
Q_i' = clamp(Q_i \cdot Multiplier_i, -2.5, 2.5)
$$

This makes personal events affect only a small proximity-weighted cluster.

### Selective Exposure Layer

Event and agent worldview are normalized:

$$
EventNorm_i = \frac{World_i}{||World_i|| + \epsilon}
$$

$$
AgentNorm_i = \frac{Exposure_i}{||Exposure_i|| + \epsilon}
$$

Alignment:

$$
Alignment_i = AgentNorm_i \cdot EventNorm_i
$$

Openness curve:

$$
OpenCurve_i = sigmoid(gain(Openness_i - 0.45))
$$

Tolerance:

$$
Tolerance_i =
base\_tolerance - (1 + openness\_factor)OpenCurve_i
$$

Suppression pressure:

$$
Pressure_i = Tolerance_i - Alignment_i
$$

Suppression:

$$
Suppression_i =
max\_suppression \cdot sigmoid(gain \cdot Pressure_i)
$$

Updated query:

$$
Q_i' = Q_i(1 - Suppression_i)
$$

High openness reduces contradiction filtering. Low openness suppresses misaligned events.

### Hybrid Attention Layer

If enabled, the agent's query is blended with the global societal mean to simulate broader context awareness:

$$
Q_i'' = (1 - hybrid\_attention\_global\_weight) \cdot Q_i' + hybrid\_attention\_global\_weight \cdot mean(Q')
$$

### Key Processing Layer

Split world tensor into positive and negative parts:

$$
K^+ = relu(K)
$$

$$
K^- = relu(-K)
$$

Threat sensitivity:

$$
ThreatSensitivity_i = 1 + Neuroticism_i \cdot threat\_sensitivity\_gain
$$

Processed key:

$$
K'_i = tanh((K^+ - K^- \cdot ThreatSensitivity_i) \cdot k\_processing\_tanh\_gain)
$$

### Relevance Layer

Importance:

$$
Importance = |Q| \cdot |K|
$$

Alignment:

$$
Alignment = Q \cdot K
$$

Relevance:

$$
R = w_{importance}Importance + w_{base}Alignment
$$

Threat amplifier:

$$
ThreatAmplifier = 1 + 1[K<0] \cdot threat\_amplifier\_gain
$$

$$
R' = R \cdot ThreatAmplifier
$$

### Temperature Layer

Temperature modulation:

$$
Temp =
1 + (C - 0.5)temp\_conscientiousness\_weight
- (N - 0.5)temp\_neuroticism\_weight
$$

clamped to `[0.5, 2.0]`.

Then:

$$
R' = R / Temp
$$

### Threshold Layer

Threshold:

$$
Threshold =
threshold\_base - (Extraversion - 0.5)threshold\_extraversion\_weight
$$

clamped to `[0, 0.3]`.

Only relevance values with absolute magnitude above the threshold survive.

### Engagement Layer

Energy:

$$
Energy_i = ||R_i||_2
$$

Engagement:

$$
Engagement_i = sigmoid(engagement\_gain(Energy_i - engagement\_threshold))
$$

The relevance vector is multiplied by engagement.

### Attention Softmax

Raw engagement energy is retained before normalization. Attention weights are:

$$
Attention_i = Softmax(5R_i)
$$

## 21. Stress-Induced Cognitive Bias

If time pressure is enabled and urgency exceeds the stress threshold:

$$
StressFactor =
sigmoid(stress\_gain(Urgency - stress\_activation\_threshold))
$$

Neuroticism amplifies the dominant attention axis:

$$
DominantValue' =
DominantValue \cdot (1 + StressFactor \cdot Neuroticism \cdot stress\_neurotic\_amplification)
$$

Low openness reduces diversity:

$$
DiversityScale =
1 - StressFactor(1-Openness)stress\_openness\_reduction
$$

Extraversion increases intensity:

$$
Intensity =
1 + StressFactor \cdot Extraversion \cdot stress\_extraversion\_boost
$$

If social conformity is enabled:

$$
BiasedAttention =
BiasedAttention + StressFactor \cdot Agreeableness \cdot conformity\_gain \cdot PopulationMean
$$

Finally attention is renormalized.

## 22. Context Construction

The context vector is:

$$
ContextScale = attention\_residual\_gain + Attention \cdot attention\_modulated\_gain
$$

$$
Context = PerceivedWorld \cdot ContextScale
$$

The implementation restores the original perceived-world norm:

$$
Context' = Context \cdot \frac{||PerceivedWorld||}{||Context||+\epsilon}
$$

Then clamps context to `[-2, 2]`.

## 23. Memory Consolidation

If agent memory is enabled, updated memory is:

$$
EffectiveDecay =
baseDecay + (1-baseDecay)(socialRehearsalFactor \cdot memory\_social\_rehearsal\_gain)
$$

clamped to at most `0.99`.

$$
Memory' = Memory \cdot EffectiveDecay + Context
$$

The orchestration layer computes social rehearsal from confidence and acting ratio:

$$
RehearsalFactor = \frac{confidence + acting\_ratio}{2}
$$

If a follow-up endogenous event exists, its context is used for memory consolidation.

## 24. Algorithmic Amplification

If enabled, simulation performs a two-pass platform-amplification flow.

Probe sample size:

$$
SampleSize = max(1, int(N \cdot algo\_sample\_size))
$$

The sample is run through cognition. Engagement-weighted attention is:

$$
EWA = Attention \cdot Engagement
$$

Average per dimension:

$$
AvgDimAttention = mean(EWA, dim=agents)
$$

The top two dimensions are selected. For each selected dimension, if its original absolute value is greater than `0.05`, it is multiplied by `algo_exaggeration_factor`.

The mutated world tensor is clamped to `[-1, 1]`, and the full society receives the amplified event.

Implementation note: the frontend exposes `algo_top_k`, `algo_min_active_value`, and `algo_max_delta`, but the backend currently hard-codes top-k as 2 and signal floor as 0.05, and it ignores those extra frontend fields.

## 25. Social Physics Engine

The social physics engine aggregates individual emotions into collective metrics.

### Influence Handling

Influence scores are converted to tensor form and saturated:

$$
StructuralWeight_i = log(1 + Influence_i)
$$

If engagement scores exist:

$$
Energy_i = \frac{Engagement_i}{mean(Engagement)+\epsilon}
$$

$$
Weight_i = StructuralWeight_i \cdot Energy_i
$$

Weights are clamped and normalized.

### Stewing Loop

For `stewing_ticks`, the engine repeatedly computes:

Objective center:

$$
Center = \sum_i Weight_i Emotion_i
$$

Local centers:

$$
LocalCenters = A \cdot CurrentEmotions
$$

or the global center if topology is absent.

Arousal:

$$
Arousal_i = ||CurrentEmotion_i||
$$

If engagement exists:

$$
ViralEnergy_i = Arousal_i \cdot \frac{Engagement_i}{mean(Engagement)+\epsilon}
$$

Alignment with local center:

$$
Alignment_i =
\frac{Emotion_i}{||Emotion_i||+\epsilon}
\cdot
\frac{LocalCenter_i}{||LocalCenter_i||+\epsilon}
$$

Validation multiplier:

$$
Validation_i = 1 + Alignment_i
$$

Viral energy:

$$
ViralEnergy_i' = ViralEnergy_i \cdot Validation_i
$$

Outrage boost:

$$
OutrageBoost_i =
1 + max\_viral\_multiplier \cdot sigmoid(outrage\_gain(ViralEnergy_i - saturation\_midpoint))
$$

Viral weights:

$$
ViralWeight_i =
\frac{Weight_i \cdot OutrageBoost_i}{\sum_j Weight_j \cdot OutrageBoost_j}
$$

Viral center:

$$
ViralCenter = \sum_i ViralWeight_i Emotion_i
$$

If valence is negative, the negative integral accumulates:

$$
NegativeIntegral += |Valence(Center)|
$$

Between ticks, emotion is updated by:

$$
NewEmotion_i =
selfRetention \cdot CurrentEmotion_i +
localInfluence \cdot LocalCenter_i +
viralInfluence \cdot ViralCenter
$$

Then arousal is restored to prevent energy loss.

## 26. Elite Center and Divergence

Elite agents are selected by top normalized weight. With `elite_percentile = 0.95`, the elite center is the weighted center of the top 5 percent.

$$
EliteCenter = \sum_{i \in elite} EliteWeight_i Emotion_i
$$

Elite-population divergence:

$$
EliteDivergence = ||EliteCenter - Center||
$$

This metric is used in endogenous event triggering and explainability.

## 27. Polarization, Bimodality, Entropy

Dispersion is:

$$
Dispersion =
\sum_i Weight_i ||Emotion_i - Center||
$$

Bimodality is measured along the dominant emotional axis. Projection:

$$
Axis = \frac{Center}{||Center||+\epsilon}
$$

$$
Projection_i = Emotion_i \cdot Axis
$$

Skewness:

$$
Skew = mean\left(\left(\frac{Projection - mean}{std}\right)^3\right)
$$

Kurtosis:

$$
Kurtosis = mean\left(\left(\frac{Projection - mean}{std}\right)^4\right)
$$

Sarle’s bimodality coefficient:

$$
BC = \frac{Skew^2 + 1}{Kurtosis + \epsilon}
$$

The implementation clamps BC to `[0, 1]` and uses it as polarization.

Entropy is computed from the clipped normalized objective center:

$$
Entropy = -\sum_d p_d log(p_d)
$$

## 28. Dominant Emotion

The dominant emotion is taken from the viral center:

$$
Dominant = argmax(ViralCenter)
$$

If max viral-center confidence is below `dominant_emotion_threshold`, the label becomes `Neutral`.

## 29. Collective Action and Granovetter Thresholds

Final arousal:

$$
Arousal_i = ||Emotion_i||
$$

Social validation:

$$
SocialValidation_i = 1 + cosine(Emotion_i, LocalCenter_i)
$$

Action cost uses personality and influence:

$$
Cost_i =
baseActionCost
- 0.1Extraversion_i
- 0.1Neuroticism_i
- 0.05log(1+Influence_i)
$$

clamped to at least `0.05`.

Individual motivation:

$$
Motivation_i = Arousal_i \cdot SocialValidation_i - Cost_i
$$

An agent is initially acting if motivated, emotional, and engaged:

$$
Motivated_i = Motivation_i > 0.1
$$

$$
Emotional_i = max(Emotion_i) >= dominant\_emotion\_threshold
$$

For personal events, engaged agents are those above half the max engagement. For general events, engaged agents are those above 10 percent of mean engagement.

If topology and Granovetter thresholds are enabled, personal thresholds are:

$$
Threshold_i =
granovetter\_threshold\_mean +
(Conscientiousness_i + Agreeableness_i - 1)granovetter\_threshold\_std
$$

clamped to `[0.01, 0.9]`.

For three iterations, agents activate if emotional, engaged, and either motivated or marginally motivated with enough acting neighbors.

Acting ratio is:

$$
ActingRatio = \frac{ActingCount}{PopulationSize}
$$

Implementation note: earlier bugs around denominator scope are covered by runtime regression tests, and the current implementation uses scoped population size.

## 30. Endogenous Events

If acting ratio exceeds `action_threshold`, the system can emit an action vector.

### Populist Uprising

Condition:

```text
elite_divergence > elite_threshold
polarization > polarization_threshold
valence < -0.2
```

Action vector:

| Dimension | Value |
| --- | ---: |
| Physical_Safety | -0.5 |
| Stability | -0.8 |
| Fairness | -0.9 |
| Freedom | +0.5 |

### Elite Policy Shift

Condition:

```text
elite_divergence > elite_threshold
norm(elite_center) > 0.3
```

Action vector:

| Dimension | Value |
| --- | ---: |
| Wealth | +0.4 |
| Fairness | -0.3 |
| Innovation | +0.6 |

### Civil Protest

Condition:

```text
polarization > polarization_threshold
dominant emotion in Anger, Disgust, Sadness
valence < -0.1
```

Action vector:

| Dimension | Value |
| --- | ---: |
| Stability | -0.6 |
| Fairness | -0.5 |
| In_Group | +0.7 |

The orchestration layer feeds the action vector back into the cognitive engine as a follow-up personal event with urgency `0.8`.

## 31. Validation Layer

The validation layer lazily loads:

```text
cardiffnlp/twitter-roberta-base-sentiment
```

It uses Hugging Face `AutoTokenizer` and `AutoModelForSequenceClassification`, on CUDA if available.

The model returns a 3-dimensional sentiment distribution. ATELIER maps its 8-dimensional emotion output to 3 sentiment buckets and compares the two distributions.

Metrics:

| Metric | Formula / Role |
| --- | --- |
| Wasserstein distance | Earth-mover distance over sentiment bucket indices. |
| KL divergence | `entropy(q_baseline, p_system)`. |
| Interpretation | `Significant Divergence` if Wasserstein > 0.2, else `Consensus`. |
| Stewing interpretation | Uses negative integral per tick. |

Stewing interpretation:

| Average negativity | Interpretation |
| ---: | --- |
| > 0.8 | Deep Structural Consequence |
| > 0.4 | Lingering Resentment |
| otherwise | Flash in the Pan |

## 32. Explainability

The explainability engine produces:

| Output | Meaning |
| --- | --- |
| shift_story | Difference between objective, viral, and elite centers. |
| tug_of_war | Polarization, entropy, bimodality, and sentiment description. |
| cognitive_drivers | Most attended world dimensions. |
| viral_dynamics | Mean/max outrage multiplier interpretation. |
| societal_structure | Influence Gini and top-10-percent narrative power. |
| demographics | Archetypes based on influence and neuroticism. |
| endogenous_events | Explanation of any autopoietic trigger. |

Demographic archetypes are:

| Archetype | Definition |
| --- | --- |
| Secure Elites | High influence, lower neuroticism. |
| Vulnerable Population | Low influence, higher neuroticism. |
| Anxious Elites | High influence, higher neuroticism. |
| Stoic Public | Low influence, lower neuroticism. |

## 33. API and Orchestration

The backend is a FastAPI app. It mounts:

| Route | Purpose |
| --- | --- |
| `GET /health` | Returns `{"status": "ok"}`. |
| `POST /simulate` | Main simulation endpoint. |
| `GET /docs` and `/docs/{slug}` | Human docs browser shell. |
| `GET /api/docs` | FastAPI Swagger/OpenAPI UI. |
| `GET /api/docs/pages` | Returns all markdown docs pages and default slug. |
| `/generated` | Static mount for generated research figures, if present. |
| `/` | Static frontend mount. |

The `/simulate` request model is:

```json
{
  "news_text": "A major employer announces AI-driven layoffs.",
  "runs": [
    {
      "seed": 42,
      "social_class": "All",
      "agent_count": 5000
    }
  ]
}
```

`RunProfile` is dynamically generated from `SimConfig`, with aliases:

| RunProfile field | SimConfig field |
| --- | --- |
| agent_count | num_agents |
| temperature | mutation_temperature |
| use_distortion | use_signal_distortion |
| use_pressure | use_time_pressure |
| use_power_law | use_power_law_influence |

The endpoint concurrently runs:

| Task | Implementation |
| --- | --- |
| LLM world-state extraction | `asyncio.create_task`. |
| RoBERTa baseline | `asyncio.to_thread`. |
| Society preparation per run | `asyncio.to_thread`. |

Society caching uses an LRU dictionary capped at 7 entries. Cache key is SHA-256 of the config dict excluding `output_dir` and `wealth_dim_idx`, with cache version `2`.

For class filtering, the orchestration layer slices metadata, exposures, personalities, affinities, memory, and sparse adjacency. Sparse adjacency is remapped to the selected index set and row-renormalized.

## 34. Frontend

The frontend is a static browser app served directly by FastAPI. It uses no separate build step.

Major UI features:

| Feature | Implementation |
| --- | --- |
| Canvas agent visualization | Each agent is a square particle. Neutral agents cluster in the center; emotional agents move to emotion-specific radial sectors. |
| Emotion palette | Joy yellow, Trust blue, Fear white-grey, Anger red, Disgust green, Sadness dark blue, Surprise orange, Anticipation purple. |
| Metrics telemetry | Majority, dominant emotion, bimodality/polarization, elite divergence slot, negative integral, active population, status. |
| Sidebar controls | Basic and researcher tabs. |
| Batch matrix | Add up to six extra experiment runs. |
| Filmstrip | Select between batch results. |
| Explainability panel | Shows reasoning, cognitive drivers, biases, shift story, viral dynamics, tug-of-war, structure, stewing, endogenous events, demographics. |
| Agent tooltip | Shows agent id, class, influence, state. |
| Agent dossier | Shows Big Five traits and cluster-level personality/class distribution. |
| History panel | Records sessions, supports JSON download and reload. |
| Docs browser | Fetches markdown pages, renders Markdown and MathJax. |

Implementation note: frontend telemetry has an elite-divergence display, but the current backend response does not return `elite_divergence` as a top-level field, so that UI value may remain unavailable unless response serialization is extended.

## 35. Documentation System

The documentation browser registers docs from the README and markdown files in `docs/`. It rewrites local markdown links into `/docs/{slug}` routes, renders Markdown server-side only as fallback, and frontend-side through `marked` plus MathJax.

Docs include:

| Document | Focus |
| --- | --- |
| README | Project overview, architecture, setup, API, testing map. |
| docs index | Reading order and subsystem map. |
| development guide | Setup, local workflow, codebase map. |
| API reference | Endpoints, request/response structure, helper functions. |
| testing guide | Full research test catalog. |
| input layer | LLM perception and world tensor extraction. |
| cognitive engine | Distortion, consensus, memory, attention, emotion projection. |
| attention context | Layer-by-layer attention gates. |
| physics engine | Aggregation, virality, stewing, polarization, endogenous events. |
| society generation | Population, topology, class assignment. |
| society evolution | Wealth, influence, ideology, mobility. |
| orchestration | Concurrency, cache, filtering, amplification, feedback. |

## 36. Raw Generated Society Data

The stored `society_data` artifacts are:

| File | Shape / Type | Key Stats |
| --- | --- | --- |
| adjacency | sparse `(10000, 10000)` float32 | 1,691,659 nonzero edges, density 0.0169, row sums approximately 1.0. |
| exposures | dense `(10000, 12)` float32 | min -1.0, mean -0.0188, std 0.3476, max 1.0. |
| personalities | dense `(10000, 5)` float32 | min 0.0005, mean 0.5018, std 0.2753, max 0.9994. |
| affinities | dense `(10000, 12)` float32 | min 0.0020, mean 0.5489, std 0.4658, max 4.7963. |
| metadata parquet | `(10000, 11)` | Agent_ID, Class, Region, Influence, Raw_Wealth, Cognitive_Bandwidth, topology/class metrics. |

Metadata numeric stats:

| Column | Min | Mean | Std | Max |
| --- | ---: | ---: | ---: | ---: |
| Influence | 0.025 | 5.611 | 10.168 | 302.320 |
| Raw_Wealth | 1398.666 | 13508.408 | 9922.863 | 287991.125 |
| Cognitive_Bandwidth | 0.100 | 0.549 | 0.196 | 1.000 |
| Topology_Degree | 11.000 | 338.332 | 294.755 | 2740.000 |
| Chamber_Wealth | 0.685 | 0.808 | 0.039 | 0.973 |
| Chamber_Influence | 0.719 | 0.829 | 0.040 | 0.984 |
| Structural_Class_Score | 0.266 | 0.608 | 0.164 | 0.957 |
| Chamber_Score | 0.708 | 0.819 | 0.039 | 0.976 |

## 37. Notebook Artifact

The notebook `society_generation_algorithm.ipynb` contains 17 cells. It documents and experiments with society-generation benchmarking, including discrete vs density-based generators, GMM-style density generation, benchmarking by runtime and peak memory, PCA/standardization visualization, structured vs bias-free generator comparisons, evolution logic, metric evaluation, and a final note that the previous probability method is better for this case, faster, less resource-intensive, and that the evolution model converges with less variance.

## 38. Research Test Suite

The suite is pytest-based and scenario-driven. It uses live defaults from the runtime schema instead of duplicating config constants. It covers contracts, topology, cognition, memory, amplification, virality, emotion mapping, validation, class segmentation, runtime regressions, and research figure generation.

Major helper files:

| File | Role |
| --- | --- |
| config schema | Defines 64 named research scenarios, live default adapters, test tensor builders, evolution override behavior. |
| metrics helpers | Gini, bimodality coefficient, graph conversion, clustering, edge cosine similarity, MAD metrics, Weisfeiler-Lehman hashes/kernels. |
| plotting utils | Paper style, palettes, figure saving, panel grid composition. |
| conftest | Adds repo root to `sys.path`. |

Test categories:

| Category | Coverage |
| --- | --- |
| Contracts | Collection integrity, RunProfile/SimConfig alignment, aliases, evolution override modes. |
| Society generation | Trait bounds, personality correlations, clustering tails, K-means trait spread. |
| Topology | Normalized adjacency, homophily, triadic closure, echo chambers, Louvain modularity. |
| Influence and inequality | Power-law influence tails, influence Gini, wealth Gini, class segmentation. |
| Cognition | Signal distortion, social consensus perception, neurotic fear divergence, selective exposure, truth refinement, personal-event localization. |
| Memory | Rehearsal decay, memory accumulation, trigger stacking. |
| Amplification and contagion | Algorithmic filter bubble, viral scaling, maximum virality cap, R0-style secondary engagement, Granovetter cascade. |
| Validation | Sentiment mapping, semantic alignment, Wasserstein/KL accuracy metrics. |
| Boundaries | Zero tensor neutrality, low-salience bounded response, dose response monotonicity. |
| Endogenous events | Stable societies do not fire macro-actions; polarized societies can. |
| Runtime | Cache isolation, fresh memory per request, small population topology, scoped acting-ratio denominator, RAM usage. |
| Figures | Summary panels, advanced visualizations, multiseed debug panels, response boundaries, trait sweeps, virality, bridge diffusion, population segmentation. |

The canonical runner is:

```bash
./run_all_tests.sh
```

Evolution modes:

```bash
./run_all_tests.sh --evolution with
./run_all_tests.sh --evolution without
./run_all_tests.sh --evolution both
```

## 39. Generated Figures and Images

The generated image artifacts include:

| Artifact Group | Meaning |
| --- | --- |
| `research_paper_summary.png` | Composite of 20 summary panels. |
| `summary_panels/01_signal_distortion.png` | Neuroticism-driven signal distortion. |
| `summary_panels/02_memory_rehearsal.png` | Memory decay with and without rehearsal. |
| `summary_panels/03_cognitive_gate.png` | Selective exposure and openness gating. |
| `summary_panels/04_algorithmic_amplification.png` | Two-pass filter bubble engagement lift. |
| `summary_panels/05_perception_consensus.png` | Social-consensus reduction in neighbor perception distance. |
| `summary_panels/06_granovetter_cascade.png` | Collective action cascade effect. |
| `summary_panels/07_echo_chambers.png` | Edge similarity under high vs low homophily. |
| `summary_panels/08_louvain_modularity.png` | Louvain modularity comparison. |
| `summary_panels/09_personality_correlations.png` | Generated personality correlation matrix. |
| `summary_panels/10_wealth_gini.png` | Baseline vs evolved wealth inequality. |
| `summary_panels/11_relative_deprivation.png` | Stronger marginalized-agent anger response. |
| `summary_panels/12_semantic_sentiment.png` | Sentiment alignment against baseline profiles. |
| `summary_panels/13_network_clustering.png` | Triadic closure clustering gain. |
| `summary_panels/14_personality_socialization.png` | Neighbor personality friction reduction. |
| `summary_panels/15_influence_tail.png` | Influence heavy-tail behavior. |
| `summary_panels/16_influence_vs_reach.png` | Structural influence and realized reach. |
| `summary_panels/17_fairness_polarization.png` | Polarization along fairness-related exposure. |
| `summary_panels/18_truth_refinement.png` | Long-term attention by skeptical agents. |
| `summary_panels/19_agent_memory.png` | Threat memory and stacking. |
| `summary_panels/20_virality_bounds.png` | Viral multiplier cap behavior. |
| `advanced_visualizations/*.png` | UMAP clusters, profiles, neuroticism spread, class mix, sentiment composition, endogenous triggers, localization, cascade size distribution. |
| `multiseed_debug/*.png` | Wealth Gini, echo similarity, influence inequality, consensus distance, trait friction, polarization across seeds. |
| `response_boundaries/*.png` | Dose response and low-salience boundaries. |
| `trait_sweeps/*.png` | Openness, extraversion, threat engagement, action cost gradients. |
| `emotion_and_bridge/*.png` | Emotion directionality and bridge diffusion. |
| `population_segmentation/*.png` | Per-dimension class response profiles. |
| `viral_scaling/*.png` | Viral scaling curve and steepest growth region. |
| `atelier_dataflow.png` | Large 2816 x 1536 dataflow diagram. |
| `research_paper.pdf` | Generated PDF version of the existing LaTeX paper. |

## 40. Numeric Results Generated by the Suite

The generated paper values report contains:

### Sentiment Mapping

| Condition | Negative | Neutral | Positive |
| --- | ---: | ---: | ---: |
| Raw | 0.500000 | 0.500000 | 0.000000 |
| Low Activity | 0.333333 | 0.666667 | 0.000000 |
| High Activity | 0.500000 | 0.500000 | 0.000000 |

Low activity adds `0.166667` neutral mass and drops negative mass by `0.166667`.

### Semantic Alignment

| Case | Wasserstein |
| --- | ---: |
| Positive world against positive baseline | 0.293700 |
| Positive world against negative baseline | 1.306300 |
| Negative world against negative baseline | 0.435600 |
| Negative world against positive baseline | 1.164400 |

Negative-minus-positive negative sentiment share is `0.365898`.

### Accuracy Metrics

| Metric | Value |
| --- | ---: |
| Matching Wasserstein | 0.546500 |
| Mismatched Wasserstein | 1.153500 |
| Wasserstein gap | 0.607000 |

### Dose Response

| Magnitude | Mean Engagement | Acting Ratio | Sentiment Valence |
| ---: | ---: | ---: | ---: |
| 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.150000 | 0.106356 | 0.777344 | -0.110000 |
| 0.300000 | 0.231087 | 0.847656 | -0.204000 |
| 0.450000 | 0.327931 | 0.882812 | -0.280000 |
| 0.600000 | 0.401527 | 0.898438 | -0.339000 |
| 0.750000 | 0.458063 | 0.921875 | -0.384000 |
| 0.900000 | 0.502383 | 0.925781 | -0.420000 |

### Low-Salience Worlds

| World | Mean Engagement | Acting Ratio | Sentiment Valence |
| --- | ---: | ---: | ---: |
| Zero | 0.000000 | 0.000000 | 0.000000 |
| Faint Threat | 0.007493 | 0.000000 | -0.019000 |
| Mixed Weak | 0.011567 | 0.000000 | -0.002000 |
| Salient Threat | 0.384224 | 0.890625 | -0.320000 |

### Emotion Directionality

| World | Dominant Emotion | Acting Ratio | Sentiment Valence |
| --- | --- | ---: | ---: |
| Prosperity | Joy | 0.886719 | 0.711000 |
| Threat | Fear | 0.921875 | -0.369000 |
| Injustice | Anger | 0.898438 | -0.361000 |

### Bridge Diffusion

| Metric | Value |
| --- | ---: |
| Acting ratio without bridge | 0.600000 |
| Acting ratio with bridge | 1.000000 |
| Acting ratio gain | 0.400000 |
| Community-B local arousal gain | 0.068000 |

### Inequality and Topology

| Metric | Value |
| --- | ---: |
| Baseline wealth Gini | 0.199135 |
| Evolved wealth Gini | 0.323614 |
| Wealth Gini delta | 0.124479 |
| Backbone clustering | 0.065414 |
| Clustering with triadic closure | 0.615779 |
| Clustering gain | 0.550366 |
| Low-homophily Louvain modularity | 0.099969 |
| High-homophily Louvain modularity | 0.285240 |
| Modularity gain | 0.185271 |

### Memory, Amplification, Virality

| Metric | Value |
| --- | ---: |
| Memory final norm gain from rehearsal | 5.592401 |
| Algorithmic amplification engagement gain | 0.044439 |
| Algorithmic amplification max world shift | 0.100000 |
| Configured viral cap | 11.000000 |
| Peak viral slope | 39.125000 |

## 41. Louvain Modularity Search

The modularity grid search reports:

| Homophily | Influence Bias | Base Connections | Triadic Prob | Modularity |
| ---: | ---: | ---: | ---: | ---: |
| 4.0 | 0.4 | 5 | 0.2 | 0.268 |
| 6.0 | 0.1 | 3 | 0.5 | 0.315 |
| 8.0 | 0.1 | 4 | 0.4 | 0.308 |
| 6.0 | 0.0 | 3 | 0.8 | 0.306 |
| 8.0 | 0.1 | 3 | 0.8 | 0.292 |
| 8.0 | 0.0 | 3 | 0.5 | 0.350 |
| 8.0 | 0.0 | 2 | 0.8 | 0.417 |

The local finding is that influence bias reduces modularity by creating cross-cluster elite bridge nodes, while high homophily, low base connections, and aggressive triadic closure produce stronger isolated echo chambers.

## 42. Engineering Dependencies

Requirements are:

```text
torch
pandas
numpy
seaborn
matplotlib
uvicorn
fastapi
pydantic
python-dotenv
transformers
pyarrow
scipy
requests
scikit-learn
python-louvain
aiohttp
```

The project requires a `GEMINI_API_KEY` for full LLM-backed simulation requests. The baseline validation model may require network access on first load to download Hugging Face weights.

## 43. Reproducibility and Execution

Local setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run app:

```bash
source .venv/bin/activate
python3 main.py
```

Run tests:

```bash
./run_all_tests.sh
```

Generate numeric paper report:

```bash
.venv/bin/pytest research_paper_tests/test_paper_values_report.py -q
```

## 44. Implementation Caveats Worth Documenting

A truly exhaustive paper should include implementation caveats rather than overstating the system.

| Caveat | Current State |
| --- | --- |
| Some config fields are reserved or inactive | `cascade_knn_k` is not materially used by current runtime logic. |
| Some frontend researcher controls are ignored by backend | Extra fields not present in `RunProfile` are ignored by Pydantic. |

## 45. Conclusion

The full codebase implements ATELIER as a layered, inspectable, deterministic social simulation framework. The key design choice is the separation of semantic parsing from social reaction. The LLM does not roleplay every agent. It produces a calibrated world tensor. The agents then react through explicit psychology, topology, memory, attention, class, and social physics.

A from-scratch research paper should therefore be written as a technical monograph rather than a short conference-style summary. It should include the world schema, matrices, tensor shapes, all formulas, topology algorithm, wealth model, evolution model, cognitive gates, memory equations, social physics equations, API behavior, UI behavior, generated artifact inventory, raw data statistics, test scenarios, numeric reports, and implementation caveats. That version would be longer, but it would preserve the actual work in the repository much more faithfully than a compressed paper.



# ATELIER: Monograph of Biases and Caveats

## Introduction
This monograph provides a comprehensive audit of the systemic biases, intentional cognitive modeling heuristics, and technical caveats within the ATELIER (Hybrid Neuro-Symbolic Multi-Agent) framework. ATELIER does not seek to produce a "bias-free" simulation; rather, it aims to mathematically formalize and inspect the specific biases that drive human social dynamics, emotional contagion, and structural polarization.

---

## 1. Systemic Design Biases (Psychological Heuristics)

The core of ATELIER’s cognitive engine is built upon a series of "Psychological Axioms" implemented as fixed matrix projections and gated layers.

### 1.1. Relative Deprivation Biases
The Relative Deprivation (RDE) layer calculates attention based on the gap between an agent's current state (exposure) and the potential state of others.
*   **Jealousy Bias**: Agents with high **Neuroticism** and low **Agreeableness** exhibit a higher "resentment factor." They attend more intensely to dimensions where their "exposure" is low compared to the population mean (perceived injustice or deficit).
*   **Protection Bias**: Agents with high **Conscientiousness** and **Extraversion** exhibit a "status/protection factor." They prioritize dimensions where their "exposure" is already high, simulating the psychological need to protect existing assets or social standing.

### 1.2. Confirmation Bias (Selective Exposure)
The **Triple-Filter-Bubble** effect is modeled via the Selective Exposure layer.
*   **Mechanism**: The system calculates the Cosine Similarity between an agent's worldview (exposures) and the incoming event tensor.
*   **Trait Gating**: High **Openness** reduces the suppression of contradictory information. Low Openness agents mathematically "block" signals that do not align with their priors, effectively enforcing a mathematically bounded confirmation bias.

### 1.3. Stress-Induced Cognitive Bias (Tunneling)
When event **Urgency** surpasses a threshold, agents experience "Cognitive Tunneling."
*   **Neurotic Amplification**: High Neuroticism agents amplify the dominant perceived threat (the dimension with the highest negative magnitude).
*   **Diversity Reduction**: High Stress combined with low Openness reduces the diversity of the attention vector, forcing the agent to hyper-fixate on a single issue.
*   **Conformity Pressure**: Agents with high **Agreeableness** drift toward the population mean under stress, sacrificing individual assessment for group safety.

### 1.4. Social Consensus Perception
Agents do not perceive the "objective" world tensor. If topology is enabled, they perceive a blend of the distorted signal and the **Local Consensus** of their neighbors. This creates a "Social Reality" bias where individuals are trapped in the perception of their immediate network.

---

## 2. Algorithmic and Structural Biases

### 2.1. Algorithmic Amplification (The Feed Bias)
The optional Two-Pass platform amplification simulates engagement-maximizing algorithms.
*   **Mechanism**: The system "probes" a sample of agents, identifies which dimensions generate the highest engagement, and then "amplifies" those dimensions in the world tensor before the full simulation run.
*   **Bias Outcome**: This creates a feedback loop that maximizes outrage and engagement, often distorting the original semantic meaning of the event to favor polarizable content.

### 2.2. Influence Bias (Topology)
In the generation of the social graph, the `influence_bias_exp` parameter controls how much "elite" or high-influence nodes act as bridges.
*   **Observation**: High influence bias reduces network modularity by forcing cross-cluster connections through elites. Conversely, zero influence bias allows for the emergence of isolated, radicalized echo chambers.

### 2.3. Viral Multipliers (Outrage Bias)
The Social Physics engine uses a non-linear sigmoid to compute "Outrage Multipliers."
*   **Bias**: Highly aroused, highly engaged agents receive up to a **10x-11x multiplier** on their emotional influence. This ensures that a small, angry minority can dominate the "Viral Center" of the society, even if the "Objective Center" remains calm.

---

## 3. Input Layer Biases (Perception)

### 3.1. LLM Calibration Bias
The system relies on a frontier LLM (e.g., Gemini) to extract the 12D world tensor.
*   **Semantic Drift**: The LLM's internal weights and training data act as the "ground truth" for what constitutes "Fairness," "Sanctity," or "Physical Safety."
*   **Frame Detection**: While the LLM is explicitly asked to detect "Corporate Spin" or "Political Framing," its ability to do so is limited by its own inherent training biases.

### 3.2. Magnitude Rubric Bias
The system uses a hardcoded magnitude rubric (e.g., 0.9 = Civilization-altering). The interpretation of "how big" an event is remains subjective to the LLM's prompt context.

---

## 4. Technical Caveats and Implementation Limits

### 4.1. Inactive Parameters
Several parameters exist in the configuration schema but are currently disconnected or have no material effect on the backend physics:
*   **`cascade_knn_k`**: Intended for localized k-nearest-neighbor cascades, but the current engine uses global or topology-based cascades.
*   **`use_social_conformity`**: While implemented in the cognitive engine, it is often toggled off by default in research scenarios to prevent over-dampening of results.

### 4.2. UI-Backend Disconnects
The frontend researcher tab includes several sliders (e.g., `algo_top_k`, `algo_min_active_value`) that are not currently mapped to the `RunProfile` Pydantic model in `main.py`. These controls are visual placeholders and use hardcoded defaults in the backend.

### 4.3. Fixed Matrices (Psychological Axioms)
The mapping from 12 World Dimensions to 8 Emotions, and the mapping from 5 Personality Traits to 12 Attention Dimensions, are **static matrices**.
*   **Caveat**: These represent a specific psychological theory (Plutchik + Big Five). They do not evolve or adapt. If the underlying psychological theory is flawed, the entire simulation’s downstream logic is affected.

### 4.4. Elite Percentile and Class Thresholds
*   **Fixedness**: Elites are strictly defined as the top 5th percentile by structural score.
*   **Class Slicing**: Class assignment occurs after topology generation and is based on a weighted sum of wealth, influence, and degree. The thresholds for "Middle Class" vs. "Working Class" are heuristic and may not reflect specific regional economic realities.

---

## 5. Statistical and Validation Caveats

### 5.1. The "Continuous Density" Assumption
The system claims to mitigate "Sampling Bias" by generating a continuous demographic spectrum rather than using scraped data.
*   **Caveat**: This merely replaces "sampling bias" with "generator bias." The society is only as diverse as the normal/log-normal distributions and Cholesky-correlated traits allow.

### 5.2. RoBERTa Validation Limits
The RoBERTa sentiment baseline is trained on Twitter data.
*   **Caveat**: Using a Twitter-trained model to validate a general-purpose social simulation introduces a "social media linguistic bias" into the validation metrics (Wasserstein distance/KL Divergence).

### 5.3. Equilibrium and Ticking
The "Stewing" loop runs for a fixed number of ticks (default 5). This is a heuristic approximation of social cooling/heating. There is no guarantee that the system has reached a mathematical equilibrium at the end of these ticks.

---

## 6. Summary of Mitigation Strategies
To counteract these caveats, the system provides:
1.  **Explainability Trace**: Every bias (jealousy, protection, stress) is explicitly named in the `shift_story` and `cognitive_drivers`.
2.  **Multi-Seed Debugging**: Research tests run across 10+ seeds to ensure observed biases are structural and not stochastic artifacts.
3.  **Dose Response Validation**: Ensuring that increasing event magnitude leads to monotonic increases in engagement and emotional intensity.

**Conclusion**: ATELIER is a laboratory for studying bias, not a black-box oracle. Users should treat the output as a "What-If" scenario driven by specific, documented psychological and algorithmic assumptions.

## 46. Algorithmic Backlash and Dual-Frame A/B Testing

To accurately predict and simulate social backlash, the system was extended with a hybrid neuro-symbolic A/B testing framework that decouples the intended "Official" narrative from the likely "Skeptical" public interpretation.

### 46.1. Dual-Frame Generation (Divergent Perception)
The Perception Layer (LLM) no longer generates a single world-state. It acts as a **Divergent Perception Engine**, outputting:
*   **Official_Frame**: The intended PR spin, best-case framing, and reputational upside (e.g., "Innovation +0.4").
*   **Skeptical_Frame**: The likely cynical, meme-driven, or critical interpretation (e.g., "Reputation -0.8", "Fairness -0.7").
*   **Backlash_Potential**: An initial LLM estimate of how likely the skeptical frame is to capture the public narrative.

### 46.2. Hybrid Backlash Prior
The system refines the LLM's potential score with a non-LLM **Backlash Prior**. This heuristic score is computed from the input text using weighted linguistic cues (e.g., "finally", "billion", "adds") and the mathematical "gap" (mean absolute difference) between the Official and Skeptical tensors. This ensures the system remains grounded in structural signals even if the LLM is overly optimistic.

### 46.3. Vanguard Trait-Based Routing
Before the global population is exposed to the event, a **Vanguard** sub-population (default 10%) is sampled for an internal A/B test.
*   **Skepticism Scoring**: Each agent receives a "Skepticism Score" calculated as a weighted sum of their Big Five traits (High Openness and Neuroticism increase skepticism; High Agreeableness and Conscientiousness reduce it).
*   **Routing**: Agents above a configurable threshold receive the **Skeptical_Frame**; those below receive the **Official_Frame**.

### 46.4. Algorithmic Narrative Flip (The Decision)
The system runs a partial cognitive pass on the vanguard and measures the resulting **Engagement Energy**. 
*   **Trigger**: If the Skeptical Frame's engagement (weighted by the combined Backlash Potential) significantly outweighs the Official Frame's engagement (exceeding the `backlash_decision_threshold`), the system detects a **Backlash Cascade**.
*   **The Flip**: The "Official" narrative is discarded, and the remaining 90% of the population is exposed exclusively to the **Skeptical_Frame**. Otherwise, the Official PR narrative propagates globally.

### 46.5. Real-World Persistence & Explainability
*   **Vanguard Memory**: The exposures experienced by the vanguard agents are committed to their individual memory tensors, ensuring that the A/B test has lasting psychological consequences for the participants.
*   **Explainability**: The engine generates a **Narrative Competition** summary, detailing the vanguard split (skeptical vs. conformist counts), the engagement energy ratio, and the sharpest points of disagreement (dimension gaps) between the competing frames.

### 46.6. Verification
The feature is validated by `test_backlash_ab_testing.py`, which confirms that highly cynical frames win the attention economy when processed by a skeptical vanguard, while routine PR holds firm in more conformist populations.
