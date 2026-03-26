# Social Physics Engine (`physics_engine.py`)

The **Social Physics Engine** is responsible for the macroscopic behavior of the simulation. It aggregates thousands of individual agent states into global societal metrics, models the spread of emotions through networks, and triggers secondary "Endogenous Events".

## 1. Socio-Emotional Aggregation

The engine calculates three distinct "Centers of Gravity" for the society:
1.  **Objective Center:** The weighted mean of all agent emotions, where weights are a combination of **Structural Influence** (e.g., wealth/status) and **Active Engagement** (how much the agent is actually paying attention).
2.  **Viral Center:** An amplified center that prioritizes high-arousal emotions (like Anger or Fear) that are being validated within local echo chambers.
3.  **Elite Center:** The emotional state of the top ~5% most influential agents, used to track **Elite–Population Divergence**.

## 2. Echo Chambers & Topological Context

If a network topology is provided (via the sparse adjacency matrix), the engine calculates a **Local Context** for every agent.
- **Mechanism:** An agent's emotional baseline is influenced by the weighted average of their neighbors' emotions.
- **Impact:** This creates "Filter Bubbles" where agents in different parts of the network can reach vastly different emotional states despite being exposed to the same global event.

## 3. Nonlinear Outrage Contagion (Virality)

The simulation models how certain emotions become "viral" through a nonlinear multiplier:
- **Arousal:** High-intensity emotions are more likely to spread.
- **Social Validation:** If an agent's emotion aligns with their local neighbors, the `validation_multiplier` increases.
- **Outrage Boost:** The final viral energy is passed through a Sigmoid function to calculate the `outrage_boost`. High outrage can amplify an agent's influence on the global state by up to 10x (`max_viral_multiplier`).

## 4. Time-Series Stewing

ATELIER does not just produce a "snapshot" reaction. It simulates the **Longitudinal Percolation** of sentiment over multiple time steps (`stewing_ticks`):
- Each tick, an agent's state is updated by blending their current state (60%), their local echo chamber (30%), and the global viral state (10%).
- This allows for the "slow-burn" of resentment or the rapid "explosion" of a viral trend.

## 5. Polarization & Stability Metrics

The engine tracks the statistical health of the society using several robust metrics:
- **Sarle's Bimodality Coefficient:** Replaces Silhouette scores to detect if the population is splitting into two opposing emotional camps. A value > 0.555 indicates high polarization.
- **Elite Divergence:** Measures the Euclidean distance between the Elite Center and the General Population. High divergence often precedes a "Populist Uprising".
- **Negative Integral:** Tracks the total "Area Under the Curve" of negative emotions over the stewing period, representing the cumulative societal trauma.

## 6. Endogenous Event Generation (Action Potential)

The framework is **autopoietic**—it can generate its own events. This is modeled using a 2-stage Action Potential:

### Stage 1: Individual Motivation
Agents calculate their internal "willingness to act" based on emotional arousal minus the "cost of action" (modulated by Extraversion and Neuroticism).

### Stage 2: Granovetter's Thresholds
Based on Granovetter's Threshold Model, agents only cross the threshold into "Action" if a critical mass of their neighbors is already acting. This simulates the "Snowball Effect" seen in protests and riots.

### Secondary Cascades
If the `acting_ratio` crosses a critical threshold, the engine triggers a new event:
- **Populist Uprising:** Triggered by high polarization and high elite divergence.
- **Civil Protest:** Triggered by high polarization.
- **Elite Policy Shift:** Triggered by high elite divergence.

These events are then fed *back* into the Cognitive Engine for a secondary cascade of reactions.
