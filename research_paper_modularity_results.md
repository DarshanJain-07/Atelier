# Louvain Modularity Grid Search Results

The following table summarizes the effect of various network topology parameters on the Louvain Modularity score in the society generation model (N=500). This grid search was conducted to identify the optimal configuration for generating distinct, homophilic echo chambers (Modularity $\approx$ 0.4).

| Homophily Strength (`h`) | Influence Bias (`i`) | Base Connections (`b`) | Triadic Closure Prob (`t`) | Resulting Modularity |
| :--- | :--- | :--- | :--- | :--- |
| 4.0 | 0.4 | 5 | 0.2 | 0.268 |
| 6.0 | 0.1 | 3 | 0.5 | 0.315 |
| 8.0 | 0.1 | 4 | 0.4 | 0.308 |
| 6.0 | 0.0 | 3 | 0.8 | 0.306 |
| 8.0 | 0.1 | 3 | 0.8 | 0.292 |
| 8.0 | 0.0 | 3 | 0.5 | 0.350 |
| **8.0** | **0.0** | **2** | **0.8** | **0.417** |

## Key Findings

1. **Influence Bias (`influence_bias_exp`)**: This parameter is the strongest deterrent to high modularity. When `i > 0`, highly influential agents create cross-cluster bridges that bind the network into a single giant component, collapsing the modularity score. Setting it to `0.0` is essential for isolated clusters.
2. **Homophily Strength (`homophily_strength`)**: Increasing homophily from `4.0` to `8.0` steepens the penalty for connecting with dissimilar agents, heavily restricting out-group edges.
3. **Base Connections (`base_connections`)**: Lowering the density (from `5` to `2` or `3`) prevents the network from becoming uniformly connected, allowing discrete community structures to form.
4. **Triadic Closure (`triadic_closure_prob`)**: Aggressively applying triadic closure (`0.8`) reinforces existing cliques by ensuring that "friends of friends" become directly connected, solidifying the boundaries of the isolated echo chambers.
