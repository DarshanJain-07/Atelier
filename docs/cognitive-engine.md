# Cognitive Engine

The **Cognitive Engine** is the "brain" of each agent. It is a high-performance PyTorch implementation that transforms the objective **World Tensor** into a subjective, internalized **Context Vector** based on an agent's unique personality, memories, and social context.

---

## 1. Subjective Perception Pipeline

The engine follows a two-stage process to simulate the "Social Construction of Reality."

### Stage 1: Individual Signal Distortion (Entropy)
Every agent receives a slightly distorted version of the world signal, simulating the "Telephone Game" effect.
- **Mechanism:** Gaussian noise is added to the 12-dimensional vector.
- **Modulation:** The magnitude ($\alpha$) is scaled by the agent's **Neuroticism** ($N$).

$$\alpha = \text{BetaSample} \cdot (1.0 + \text{gain}_{dist} \cdot N) \cdot \text{max\_noise}$$
$$Perceived_{Indiv} = WorldTensor + (\alpha \cdot \text{Noise})$$

### Stage 2: Socially Constructed Reality (Consensus)
Individual perceptions are refined by the agent's local social network (neighbors).
- **Mechanism:** The distorted signal is blended with the average perception of neighbors.

$$Perceived_{Social} = (1 - g_{soc}) \cdot Perceived_{Indiv} + g_{soc} \cdot \text{LocalNeighborMean}$$

---

## 2. The Attention Pipeline

Before an agent reacts, the signal passes through a modular attention pipeline that determines which dimensions the agent "cares about" (Relevance). This includes complex cognitive filtering such as:
- **RDE Layer**: Filtering based on relative deprivation/enhancement.
- **Logic Consistency**: Detecting discrepancies (Skepticism).
- **Selective Exposure**: Blocking contradicting information (Filter Bubbles).

*See the [Attention Context (Gates)](./attention-context.md) documentation for full layer-by-layer architectural details.*

---

## 3. Stress-Induced Cognitive Bias (Tunneling)

When the `Urgency` of an event is high, the engine triggers **Cognitive Tunneling**. This is modeled via a smooth stress factor:

$$StressFactor = \sigma(\text{gain}_{stress} \cdot (Urgency - \text{threshold}))$$

### Effects:
- **Neurotic Amplification**: Fixating on the most dominant/threatening dimension.
  $$Attention_{dom} = Attention_{dom} \cdot (1.0 + StressFactor \cdot N \cdot 1.5)$$
- **Cognitive Closure**: Reducing diversity of thought (lowering Openness).
  $$DiversityScale = 1.0 - StressFactor \cdot (1 - O) \cdot 0.5$$
- **Social Conformity**: Pulling attention weights toward the population mean.

---

## 4. Memory & Trauma Mechanics

The engine maintains a persistent memory tensor, enabling long-term sociological behavior.

### I. Desensitization (Fatigue)
Repeated exposure to similar events reduces their emotional impact over time.
$$\text{Fatigue} = 1.0 - \exp(-\text{Alignment} \cdot \text{gain}_{fatigue})$$

### II. Trigger Stacking (Sensitization)
Cumulative past stress amplifies the reaction to *new* threats.
$$\text{TriggerBoost} = \log(1 + \text{TotalPastStress}) \cdot \text{gain}_{trigger}$$

### III. Social Rehearsal (Consolidation)
Memory decay is slowed down if an event is "globally viral" or rehearsed by the network.
$$\text{EffectiveDecay} = \text{BaseDecay} + (1.0 - \text{BaseDecay}) \cdot (\text{Virality} \cdot \text{gain}_{rehear})$$

---

## 5. Key Parameters & Outcome Impacts

| Parameter | Impact on Results |
|---|---|
| `distortion_neurotic_gain` | How much Neuroticism amplifies signal noise/misinterpretation. |
| `perception_social_consensus_gain` | The strength of "Socially Constructed Reality" (blending with neighbors). |
| `stress_activation_threshold` | The Urgency level at which cognitive tunneling/panic begins. |
| `memory_decay_rate` | How quickly agents "forget" or move past previous events. |
| `memory_trigger_stacking_gain` | Determines how much past trauma amplifies current threat perception. |
