# Society Evolution (`society_evolution.py`)

The **Society Evolution** module handles the long-term, multi-generational shifts in the simulation. It transforms a static population into a dynamic society where wealth, influence, and ideology evolve through inheritance, reinvestment, and cultural drift.

---

## 1. Evolution Cycle

The evolution process runs for a set number of generations ($T$), applying the following transformations in sequence:

1.  **Inheritance & Redistribution**: Passing down wealth and taxation for public goods.
2.  **Reinvestment Cycles**: Wealth growth driven by social influence.
3.  **Economic Shocks**: Rare, impactful global events.
4.  **Social Mobility**: Random reshuffling of status.
5.  **Ideological Drift**: Cultural shifts toward global or elite means.
6.  **Class Reassignment**: Updating social hierarchies based on power scores.
7.  **Idiosyncratic Drift**: Deterministic personality divergence.

---

## 2. Wealth & Status Dynamics

### I. Inheritance Tax & Basic Income (UBI)
Wealth is not fully dynastic; a portion is "taxed" and redistributed equally to prevent total economic stagnation.

$$Inherited = ParentWealth \cdot inheritance\_fraction$$
$$Redistributed = \frac{\sum (ParentWealth - Inherited)}{N_{agents}}$$

### II. Reinvestment & Social Leverage
Wealthy and influential agents can "leverage" their status for higher returns.
$$Returns = r_{base} + r_{influence} \cdot \left(\frac{Influence}{\bar{Influence}}\right) + \epsilon$$

---

## 3. Cultural Hegemony & Ideology

### I. The Attractor Model
Agents' world views drift toward a "Cultural Attractor."
- **Cultural Consensus**: Drift toward the global mean (default).
- **Cultural Hegemony**: There is a 5% chance (`elite_influence_drift_chance`) that society drifts toward the **Elite Class** mean, simulating the influence of the powerful over general culture.

### II. Alienation & Repulsion
If an agent's similarity to the societal mean falls below a certain `repulsion_threshold`, they stop being attracted and start being **repelled** from the mainstream.

$$Drift = \begin{cases} +(Target - Agent) \cdot gain & \text{if Similarity } \geq Threshold \\ -(Target - Agent) \cdot gain_{repel} & \text{if Similarity } < Threshold \end{cases}$$

---

## 4. Hierarchy & Class Structures

### Power Score & Reassignment
Classes (Underclass, Working, Middle, Upper Middle, Elite) are dynamic. They are reassigned using a **Power Score** based on relative Wealth, Influence, and Personality.

$$PowerScore = 0.5 \cdot Wealth_{norm} + 0.4 \cdot Influence_{norm} + 0.1 \cdot Traits_{mean}$$

Agents are assigned using a **Softmax** probability based on their proximity to 5 structural class centers (0.1, 0.3, 0.5, 0.75, 0.95).

---

## 5. Key Evolution Parameters

| Parameter | Impact on Results |
|---|---|
| `evolution_generations` | Total time-steps for the generational simulation. |
| `inheritance_fraction` | Ratio of wealth passed down (1.0 = total dynasty, 0.0 = total redistribution). |
| `elite_influence_drift_chance` | Likelihood that global ideology follows the elite class rather than the average. |
| `shock_frequency` | Probability of a major negative/positive wealth event occurring this generation. |
| `mobility_rate` | Percent of the population that "swaps" positions, representing systemic social mobility. |
| `class_temperature` | Controls the "fluidity" of class reassignment. High temperature = more noise in class status. |
