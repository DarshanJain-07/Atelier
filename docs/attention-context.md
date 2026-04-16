# Attention Context & Cognitive Gates

The **Attention Context** is a modular pipeline within the Cognitive Engine that determines the **Internalized Relevance** of a world event for an individual agent. It implements a series of "Cognitive Gates" that filter, amplify, or suppress the 12-dimensional World Tensor based on an agent's Big Five personality traits and existing exposures.

---

## 1. The Attention Pipeline Architecture

The pipeline transforms the objective **World Tensor** ($K$) into an agent-specific **Relevance Vector** ($R$) through a sequence of differentiable layers.

### I. RDE Layer (Relative Deprivation/Enhancement)
Calculates initial sensitivity based on what the agent lacks (**Gaps**) and what they possess (**Assets**).

- **Jealousy Bias**: Driven by Neuroticism ($N$) and low Agreeableness ($A$).
  $$Jealousy = N \cdot gain_{jealousy} + (1 - A) \cdot gain_{resentment}$$
- **Protection Bias**: Driven by Conscientiousness ($C$) and Extraversion ($E$).
  $$Protection = C \cdot gain_{protection} + E \cdot gain_{status}$$

$$RDE\_Sensitivity = (Gaps \cdot Jealousy) + (Assets \cdot Protection)$$

### II. Personality Query Layer
Modulates the "Query Vector" ($Q$) by prioritizing dimensions that align with the agent's core personality traits (defined in `PERSONALITY_QUERY_MATRIX`).
- **Openness**: Prioritizes *Innovation* and *Freedom*.
- **Conscientiousness**: Prioritizes *Wealth* and *Reputation*.
- **Agreeableness**: Prioritizes *Fairness* and *Care*.

### III. Logic Consistency (The Skepticism Gate)
Detects discrepancies between **Short-Term** ($ST$) and **Long-Term** ($LT$) impacts. High Openness/Conscientiousness agents are more likely to detect "too good to be true" flaws.

$$LogicGap = |ST - LT|$$
$$DetectionProb = \sigma(SkepticismTrait \cdot (LogicGap - Threshold))$$

- **Impact**: If a gap is detected, attention to the Short-Term signal is suppressed by up to 85%, and Long-Term focus is boosted.

### IV. Selective Exposure (The Filter Bubble Gate)
Implements **Confirmation Bias**. If an event fundamentally contradicts an agent's existing world view (measured via Cosine Similarity between their `exposures` and the `World Tensor`), they may block the information entirely.

- **Trigger**: Activated if $CosineSimilarity(Worldview, Event) < ToleranceThreshold$.
- **Modulation**: Agents with high **Openness** have a lower (more tolerant) threshold.

### V. Hybrid Attention Layer
Blends the individual agent's specific query with the societal mean query to simulate awareness of the broader context or "zeitgeist" (inspired by local/global attention mechanisms).

$$Q_{hybrid} = (1 - \gamma) \cdot Q_{local} + \gamma \cdot Q_{global}$$

- **Modulation**: The blending ratio ($\gamma$) is controlled by `hybrid_attention_global_weight`.

---

## 2. Key Processing & Threat Sensitivity

The engine processes the World Tensor ($K$) into a subjective "Key" vector.

- **Neurotic Amplification**: Agents with high **Neuroticism** ($N$) are hypersensitive to negative signals (threats).
  $$ThreatSensitivity = 1.0 + N \cdot gain_{threat}$$
  $$K_{subjective} = K_{positive} - (K_{negative} \cdot ThreatSensitivity)$$

---

## 3. Global Relevance & Engagement

The final relevance is computed by combining the Query ($Q$) and Key ($K$).

### I. Relevance Equation
$$Relevance = (w_{imp} \cdot |Q| \cdot |K| + w_{base} \cdot Q \cdot K) \cdot ThreatAmplifier$$
*Where $ThreatAmplifier$ adds a final boost if the signal is negative ($K < 0$).*

### II. Engagement Gating
A final sigmoid gate determines if the agent "engages" with the event at all. If the total energy of the relevance vector is below the `engagement_threshold`, the agent ignores the event.

$$Engagement = \sigma(gain_{eng} \cdot (||Relevance|| - Threshold_{eng}))$$

---

## 4. Key Parameters & Outcome Impacts

| Parameter | Impact on Results |
|---|---|
| `skepticism_gain` | Determines how effectively agents see through short-term "PR spin" versus long-term consequences. |
| `logic_gap_threshold` | The level of discrepancy between ST/LT required to trigger a skeptical response. |
| `use_selective_exposure` | Toggles the "Filter Bubble" effect. If `True`, agents with low Openness will block opposing views. |
| `threat_sensitivity_gain` | Controls how much Neuroticism amplifies the perception of negative events. |
| `engagement_threshold` | Sets the minimum "importance" required for an agent to notice an event. |
