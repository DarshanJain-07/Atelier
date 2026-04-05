# Perception Layer

The **Perception Layer** is the entry point of the ATELIER framework. It serves as a **Neuro-Symbolic bridge**, translating unstructured natural language (news, policies, social media) into a structured 12-dimensional **World Tensor** and metadata that drives the multi-agent simulation.

## 1. Input Parameters

| Parameter | Type | Description |
|---|---|---|
| `user_input` | `string` | The natural language text describing an event, policy, or news item. |

---

## 2. The Predictive Engine (LLM)

The layer utilizes the `gemini-3-flash-preview` model. Unlike standard sentiment analysis, it is instructed to act as a **Predictive World Model**, piercing PR spin to identify actual material outcomes.

### The Magnitude Rubric (Calibration)
To maintain statistical consistency, the LLM adheres to a strict calibration scale:

| Score | Category | Examples |
|---|---|---|
| **0.0** | Neutral | No measurable impact. |
| **±0.1 - 0.2** | Routine / Minor | Local news, minor tech updates, routine speeches. |
| **±0.3 - 0.5** | Significant | National protests, major bankruptcies, significant elections. |
| **±0.6 - 0.8** | Crisis / Boom | Global financial crashes, pandemics, outbreaks of war. |
| **±0.9 - 1.0** | Civilization Altering | AGI Singularity, Nuclear War, Alien Contact. |

### Logic: "Piercing the Spin"
The engine identifies **framing tactics**:
- **Corporate Spin:** e.g., "Right-sizing for efficiency" is mapped to **Negative Wealth** and **Negative Stability** for the workforce.
- **Power Dynamics:** Recognizes condescending language or threats to sovereignty, mapping them to **Negative Freedom** and **Negative Fairness**.

---

## 3. Output Parameters (The World State)

### A. World Tensor (1 x 12)
A PyTorch tensor representing the objective material impact across 12 sociological dimensions:
1.  **Wealth**: Macro-economic resources/capital.
2.  **Physical_Safety**: Bodily harm or imminent threat.
3.  **Stability**: Social order and predictability.
4.  **Reputation**: Prestige and social standing.
5.  **Fairness**: Justice and lack of exploitation.
6.  **In_Group**: Internal cohesion and tribal loyalty.
7.  **Innovation**: Progress and novelty.
8.  **Freedom**: Autonomy and lack of coercion.
9.  **Sanctity**: Purity and cultural taboos.
10. **Care**: Empathy and protection of the vulnerable.
11. **Short_Term**: Immediate, transient effects.
12. **Long_Term**: Lasting, systemic shifts.

### B. Urgency (0.0 to 1.0)
Represents time pressure.
- **0.0:** Historical trends or slow-moving events.
- **1.0:** Immediate "Fight or Flight" (e.g., active shooter, imminent missile strike).

### C. Is_Personal (Boolean)
- **True:** Only if the text uses first-person pronouns ("I", "Me", "My") implying the user is the direct target.

---

## 4. Downstream Impact & Equations

The outputs of the Perception Layer trigger specific deterministic logic in the **Simulation Layer**.

### I. Cognitive Tunneling (Stress Bias)
High **Urgency** scores trigger a shift in how agents process information.
If $Urgency > threshold$ (default 0.3):

$$StressFactor = \sigma(gain \cdot (Urgency - threshold))$$

- **Neuroticism (N)**: Amplifies the dominant interpretation.
  $$Amplification = 1.0 + StressFactor \cdot N \cdot 1.5$$
- **Openness (O)**: Reduces cognitive diversity and skepticism.
  $$DiversityScale = 1.0 - StressFactor \cdot (1 - O) \cdot 0.5$$
- **Agreeableness (A)**: Drives social conformity (if enabled).
  $$Conformity = StressFactor \cdot A \cdot gain$$

### II. Emotional Projection
The 12D World Tensor is projected into 8 Plutchik emotions using the **Psychological Axiom Matrix** ($P$) defined in `schema.py`:

$$Emotions = WorldTensor \times P$$

*Example Axioms:*
- **Negative Safety** $\rightarrow$ **Fear**
- **Negative Fairness** $\rightarrow$ **Anger**
- **Negative Sanctity** $\rightarrow$ **Disgust**
- **Positive Innovation** $\rightarrow$ **Surprise / Anticipation**

### III. Signal Distortion (Entropy)
The agent's **Neuroticism** determines how much noise is added to the objective World Tensor:

$$DistortedSignal = WorldTensor + (\alpha \cdot Noise)$$

Where $\alpha$ (distortion magnitude) is calculated as:
$$\alpha = BaseAlpha \cdot (1.0 + Neurotic Gain \cdot Neuroticism)$$
