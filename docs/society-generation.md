# Society Generation & Evolution

The **Society Generation** module (Step 2 in the ATELIER pipeline) initializes the population, their network topology, and their socioeconomic trajectories. It follows a **2-Stage Refinement Strategy**, ensuring that individual traits (Biology/Psychology) are modulated by the network (Sociology/Culture).

---

## 1. 2-Stage Wealth Engine (`generate_network_wealth`)

Wealth is not an isolated attribute; it is a "Networked Realization" of potential.

### Stage 1: Latent Individual Potential
Initial capital is generated based on individual **Merit** (Conscientiousness) and **Innate Influence**.

$$SeedPotential = 500 \cdot (Influence^{0.7}) + (Conscientiousness \cdot 2000)$$

### Stage 2: Network Realization & Cluster Lift
Wealth is amplified by the agent's position in the topology.
- **Clustered Lift**: Agents "pull up" their neighbors. If you are connected to high-wealth individuals, your own wealth rises.
- **Social Capital**: A non-linear multiplier based on **In-Degree** (number of followers).

$$SocialCapitalMult = 1.0 + 0.4 \cdot \sqrt{InDegree}$$
$$FinalWealth = (SeedPotential \cdot 0.3 + ClusterLift \cdot 0.7) \cdot SocialCapitalMult$$

---

## 2. 2-Stage Topology Construction (`create_topology`)

The social network is built to reflect both global influence and local community cliques.

### Stage 1: Structural Backbone (Homophily)
Initial edges are formed based on **Trait Similarity** and **Preferential Attachment**.
- Agents prefer connecting to those with similar Big Five profiles (Homophily).
- High-influence agents attract more connections.

### Stage 2: Community Cohesion (Triadic Closure)
The engine performs iterative **Triadic Closure** (A knows B, B knows C $\rightarrow$ A meets C).
- **Impact**: Creates high-density local clusters (cliques) essential for modeling **Echo Chambers**.
- **Parameter**: `triadic_closure_prob` (default 0.2) controls the density of these cliques.

---

## 3. Personality Socialization (`generate_society`)

Agent personalities undergo a "Nurture" phase after initialization.

- **Stage 1 (Nature)**: Initial assignment via Cholesky correlation matrix.
- **Stage 2 (Nurture)**: Traits drift toward the local network mean.

$$Personality_{new} = (1 - gain) \cdot Personality_{initial} + gain \cdot LocalMean$$

*Where $gain$ is the `personality_socialization_gain` (default 0.05).*

---

## 4. Generational Evolution

For long-term, multi-generational shifts, the framework utilizes the **Society Evolution** module. This handles inheritance, reinvestment cycles, ideological drift (Hegemony/Repulsion), and dynamic social mobility.

*See the [Society Evolution](./society-evolution.md) documentation for full details.*

---

## 5. Key Parameters & Outcome Impacts

| Parameter | Impact on Results |
|---|---|
| `num_agents` | Controls the statistical resolution and computational load. |
| `mutation_temperature` | Increases "Entropy." High values create more radical outliers and diverse fringe groups. |
| `homophily_strength` | Higher values create stronger, more isolated Echo Chambers. |
| `inheritance_fraction` | Controls wealth inequality over time. Low values promote mobility; high values entrench dynasties. |
| `mobility_rate` | The percentage of agents who randomly swap wealth/influence, simulating "The American Dream" or sudden ruin. |
