import math
import os

import numpy as np
import pandas as pd
import torch

from schema import (
    DIMENSION_INDICES,
    DIMENSIONS,
    PERSONALITY_CORRELATIONS,
    SimConfig,
)
from society_evolution import SocietyEvolution


# ================================
# Mutation Logic
# ================================
def apply_random_mutations(exposures, personality_logits, temperature, seed):
    if temperature <= 0.0:
        return exposures, personality_logits

    rng = torch.Generator()
    rng.manual_seed(seed + 999)

    n_agents = exposures.shape[0]
    num_dims = len(DIMENSIONS)

    mutation_prob = temperature
    mutant_mask = torch.rand(n_agents, generator=rng) < mutation_prob
    mutant_indices = torch.where(mutant_mask)[0]

    if len(mutant_indices) == 0:
        return exposures, personality_logits

    num_changes = math.ceil(3 * temperature)

    for _ in range(num_changes):
        col_indices = torch.randint(0, num_dims, (len(mutant_indices),), generator=rng)
        # Use normal distribution for mutations to preserve bell curve
        random_values = torch.clamp(
            torch.randn(len(mutant_indices), generator=rng) * 0.4, -1.0, 1.0
        )
        exposures[mutant_indices, col_indices] = random_values

    # Apply additive noise to personality logits
    for _ in range(num_changes):
        col_indices = torch.randint(0, 5, (len(mutant_indices),), generator=rng)
        noise = torch.randn(len(mutant_indices), generator=rng) * 0.5
        personality_logits[mutant_indices, col_indices] += noise

    # ---- Radical Outliers ----
    radical_prob = 0.05 * temperature
    radical_mask = torch.rand(n_agents, generator=rng) < radical_prob
    radical_indices = torch.where(radical_mask)[0]

    if len(radical_indices) > 0:
        for _ in range(2):
            col_indices = torch.randint(0, 5, (len(radical_indices),), generator=rng)
            radical_values = torch.randn(len(radical_indices), generator=rng) * 2.5
            personality_logits[radical_indices, col_indices] = radical_values

    return exposures, personality_logits


# ================================
# NETWORK SYNERGY WEALTH ENGINE
# ================================
def generate_network_wealth(
    influence_scores,
    personality_logits,
    adjacency_matrix,
    temperature,
    seed,
):
    """
    2-Stage Network Synergy Model:
    1. Latent Potential: Based on individual Influence + Merit (Conscientiousness).
    2. Network Realization: Wealth diffuses via KNN topology (Clustered Lift) 
       and is amplified by Social Capital (In-Degree).
    """
    from scipy.stats import pareto
    
    np.random.seed(seed + 123)
    n = len(influence_scores)

    # --- Stage 1: Latent Individual Potential ---
    # Conscientiousness (Index 1) drives base productivity
    consc = torch.sigmoid(personality_logits[:, 1]).numpy()
    
    # Power Law for potential, but dampened (exponent 0.7) to allow network effect to dominate
    # Base: 500 units + up to 2000 from Merit
    seed_potential = 500.0 * (influence_scores ** 0.7) + (consc * 2000.0)

    # --- Stage 2: Network Realization & Cluster Lift ---
    if adjacency_matrix is not None:
        w_seed_tensor = torch.tensor(seed_potential, dtype=torch.float32).unsqueeze(1)
        
        # A. Clustered Lift (Neighbor Average)
        # Connected people pull each other up. If connected to a whale, you rise.
        cluster_lift = torch.sparse.mm(adjacency_matrix, w_seed_tensor).squeeze().numpy()
        
        # B. Social Capital (In-Degree Multiplier)
        adj_coalesced = adjacency_matrix.coalesce()
        indices = adj_coalesced.indices()
        in_degrees = torch.bincount(indices[1], minlength=n).float().numpy()
        
        # Non-linear boost for being 'followed' (Social Capital)
        social_capital_mult = 1.0 + 0.4 * np.sqrt(in_degrees)
        
        # C. Synthesis: Blend Individual Potential (30%) with Network Cluster (70%)
        # This creates 'Tangled Wealth' where organizations/echo-chambers rise together.
        wealth = (seed_potential * 0.3 + cluster_lift * 0.7) * social_capital_mult
    else:
        wealth = seed_potential

    # --- Elite Cluster Synergy (Multiple Billionaires) ---
    # Instead of one outlier, we inject Pareto wealth into the top 10% of the hierarchy.
    # Because wealth is clustered, if one person in an elite cluster gets this, 
    # the 'Lift' logic in the next generation or simulation cycle would spread it.
    # Here we simulate the existing 'Legacy' wealth of clusters.
    alpha = 2.0 - (temperature * 0.5)
    legacy_injection = (pareto.rvs(alpha, size=n) + 1.0) * 10000.0 * temperature
    
    # Gate legacy by network position (Top 15% by current synergy wealth)
    ranks = np.argsort(np.argsort(wealth)) / (n - 1)
    legacy_gate = 1.0 / (1.0 + np.exp(-25.0 * (ranks - 0.85)))
    
    wealth += (legacy_injection * legacy_gate)

    # Final Clamping & Subsistence Floor
    wealth = np.maximum(wealth, 500.0)
    wealth = np.nan_to_num(wealth, nan=500.0, posinf=1e8)

    return wealth


# ================================
# Main Generator (Topology-First)
# ================================
def apply_triadic_closure(config: SimConfig, adj: torch.Tensor):
    """
    Stage 2: Community Cohesion via Iterative Triadic Closure.
    Forms new edges between neighbors-of-neighbors (A->B, B->C => A->C).
    """
    if not getattr(config, "triadic_closure_prob", 0.0) > 0:
        return adj

    N = config.num_agents
    prob = config.triadic_closure_prob
    iterations = getattr(config, "triadic_closure_iterations", 1)
    
    current_adj = adj.coalesce()

    for _ in range(iterations):
        # 1. Find neighbors-of-neighbors using sparse matrix multiplication
        # A_2[i, j] > 0 if there is a path i -> k -> j
        # We use a simplified version: just the structure, not the weights
        indices = current_adj.indices()
        vals = torch.ones_like(current_adj.values())
        binary_adj = torch.sparse_coo_tensor(indices, vals, size=(N, N)).coalesce()
        
        # Paths of length 2
        paths_2 = torch.sparse.mm(binary_adj, binary_adj).coalesce()
        
        p2_indices = paths_2.indices()
        p2_values = paths_2.values()
        
        # 2. Filter: Remove self-loops and existing edges
        # We only want NEW edges
        mask_self = p2_indices[0] != p2_indices[1]
        
        # To check existing edges efficiently, we can use a temporary sparse mask
        # or just combine and let coalesce() handle duplicates (but we want to exclude them)
        # A better way: use a small random sample of paths_2
        
        # Sample based on triadic_closure_prob
        sample_mask = torch.rand(len(p2_values)) < prob
        valid_mask = mask_self & sample_mask
        
        if not valid_mask.any():
            break
            
        new_indices = p2_indices[:, valid_mask]
        # Assign a base weight for these new 'social' connections
        # We use the average weight of existing connections to keep it balanced
        new_values = torch.full((new_indices.shape[1],), current_adj.values().mean())
        
        # 3. Merge with original backbone
        combined_indices = torch.cat([current_adj.indices(), new_indices], dim=1)
        combined_values = torch.cat([current_adj.values(), new_values])
        
        current_adj = torch.sparse_coo_tensor(combined_indices, combined_values, size=(N, N)).coalesce()
        
        # Cap connections to prevent exploding density
        # (Optional, but safe for performance)
        if current_adj._nnz() > N * config.max_connections:
            break

    return current_adj

def create_topology(
    config: SimConfig,
    exposures: torch.Tensor,
    personalities: torch.Tensor,
    influence_scores: np.ndarray,
):
    """
    2-Stage Topology Construction:
    1. Structural Backbone: Preferential Attachment and Homophily.
    2. Community Cohesion: Iterative Triadic Closure (Neighbors-of-Neighbors).
    """
    N = config.num_agents
    inf_mean = np.mean(influence_scores)
    k_array = np.clip(
        (influence_scores / inf_mean) * config.base_connections,
        1,
        config.max_connections,
    ).astype(int)

    features = torch.cat([exposures, personalities], dim=1)
    features_norm = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)

    batch_size = 1000
    indices_list = []
    values_list = []
    device = exposures.device
    features_norm = features_norm.to(device)

    inf_tensor = torch.tensor(influence_scores, dtype=torch.float32, device=device)
    inf_tensor = inf_tensor / inf_tensor.mean()

    # --- Stage 1: Structural Backbone ---
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch_size_actual = end - i
        batch_features = features_norm[i:end]
        sim = torch.mm(batch_features, features_norm.T)
        sim = torch.pow((sim + 1.0) / 2.0, getattr(config, "homophily_strength", 2.0))
        prob_matrix = sim * inf_tensor.unsqueeze(0)

        batch_indices = torch.arange(batch_size_actual, device=device)
        global_indices = batch_indices + i
        prob_matrix[batch_indices, global_indices] = 0.0

        batch_k = k_array[i:end]
        max_k = int(np.max(batch_k))
        if max_k == 0: continue

        prob_matrix = prob_matrix + 1e-9
        sampled_indices = torch.multinomial(prob_matrix, max_k, replacement=False)
        sampled_vals = torch.gather(sim, 1, sampled_indices)

        range_tensor = torch.arange(max_k, device=device).unsqueeze(0)
        k_tensor = torch.tensor(batch_k, device=device).unsqueeze(1)
        mask = range_tensor < k_tensor

        valid_src = torch.arange(i, end, device=device).unsqueeze(1).expand(-1, max_k)[mask]
        valid_dst = sampled_indices[mask]
        valid_val = sampled_vals[mask]

        indices_list.append(torch.stack([valid_src, valid_dst]))
        values_list.append(valid_val)

    if not indices_list: return None

    indices_tensor = torch.cat(indices_list, dim=1)
    values_tensor = torch.cat(values_list)
    backbone_adj = torch.sparse_coo_tensor(indices_tensor, values_tensor, size=(N, N)).to(device)

    # --- Stage 2: Community Cohesion (Triadic Closure) ---
    print(f"Applying Triadic Closure (Stage 2)...")
    final_adj = apply_triadic_closure(config, backbone_adj)

    # --- Final Normalization ---
    final_adj = final_adj.coalesce()
    row_indices = final_adj.indices()[0]
    dense_sums = torch.sparse.sum(final_adj, dim=1).to_dense()
    dense_sums = torch.clamp(dense_sums, min=1e-8)
    
    normalized_values = final_adj.values() / dense_sums[row_indices]

    normalized_sparse_adj = torch.sparse_coo_tensor(
        final_adj.indices(), normalized_values, size=(N, N)
    ).coalesce()
    
    return normalized_sparse_adj


def generate_society(config: SimConfig):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    print(f"Generating {config.num_agents} Agents via Clustered Synergy Model...")

    num_dims = len(DIMENSIONS)
    wealth_idx = DIMENSION_INDICES["Wealth"]
    num_personalities = 5
    total_dims = num_dims * 2 + num_personalities

    # 1. Initialize Continuous Traits (Wealth=0 for now)
    traits = torch.randn(config.num_agents, total_dims) * config.initial_trait_std_dev
    exposures = traits[:, :num_dims]
    exposures[:, wealth_idx] = 0.0 # Placeholder

    # 2. Correlated Personalities
    raw_personalities = traits[:, num_dims : num_dims + num_personalities]
    try:
        standard_noise = raw_personalities / config.initial_trait_std_dev
        jitter = torch.eye(5) * 1e-4
        L = torch.linalg.cholesky(PERSONALITY_CORRELATIONS + jitter)
        correlated_noise = torch.matmul(standard_noise, L.T)
        raw_personalities = correlated_noise * config.initial_trait_std_dev
    except Exception as e:
        print(f"Warning: Cholesky failed ({e}).")

    raw_affinities = traits[:, num_dims + num_personalities :]
    exposures, raw_personalities = apply_random_mutations(
        exposures, raw_personalities, config.mutation_temperature, config.seed
    )
    personalities = torch.sigmoid(raw_personalities)

    # 3. Influence
    influence_scores = np.random.lognormal(
        mean=1.0, sigma=0.5 + config.mutation_temperature, size=config.num_agents
    )
    if getattr(config, "use_power_law_influence", False):
        alpha = 1.16
        pareto_multiplier = (np.random.pareto(alpha, config.num_agents) + 1) * 2.0
        influence_scores *= pareto_multiplier

    # 4. Topology Generation (Based on Traits + Influence)
    adjacency_matrix = None
    if getattr(config, "use_network_topology", True):
        print("Generating Network Topology...")
        adjacency_matrix = create_topology(config, exposures, personalities, influence_scores)

    # 5. Clustered Wealth Generation (Stage 2 Synergy)
    wealth_values = generate_network_wealth(
        influence_scores,
        raw_personalities,
        adjacency_matrix,
        config.mutation_temperature,
        config.seed,
    )

    # 6. Normalize Wealth for Exposures
    log_wealth = np.log1p(wealth_values)
    min_w, max_w = np.min(log_wealth), np.max(log_wealth)
    if max_w > min_w:
        wealth_normalized = 2.0 * ((log_wealth - min_w) / (max_w - min_w)) - 1.0
    else:
        wealth_normalized = np.zeros_like(log_wealth)
    exposures[:, wealth_idx] = torch.tensor(wealth_normalized).float()

    # Final Touches
    non_wealth_mask = torch.ones(num_dims, dtype=torch.bool)
    non_wealth_mask[wealth_idx] = False
    exposures[:, non_wealth_mask] = torch.tanh(exposures[:, non_wealth_mask])
    
    cognitive_bandwidth = torch.clamp(torch.randn(config.num_agents, 1) * 0.2 + 0.55, min=0.1, max=1.0)
    positive_affinities = torch.clamp(torch.abs(raw_affinities), min=0.01)
    normalized_affinities = positive_affinities / positive_affinities.sum(dim=1, keepdim=True)
    affinities = normalized_affinities * cognitive_bandwidth

    df_metadata = pd.DataFrame({
        "Agent_ID": range(config.num_agents),
        "Class": ["Agent"] * config.num_agents,
        "Region": ["Global"] * config.num_agents,
        "Influence": np.round(influence_scores, 3),
        "Raw_Wealth": np.round(wealth_values, 3),
        "Cognitive_Bandwidth": np.round(cognitive_bandwidth.squeeze().numpy(), 3),
    })

    df_metadata.to_parquet(f"{config.output_dir}/metadata.parquet")
    torch.save(exposures, f"{config.output_dir}/exposures.pt")
    torch.save(personalities, f"{config.output_dir}/personalities.pt")
    torch.save(affinities, f"{config.output_dir}/affinities.pt")
    if adjacency_matrix is not None:
        torch.save(adjacency_matrix, f"{config.output_dir}/adjacency.pt")

    print(f"Society Generated in '{config.output_dir}' (Network Synergy Model)")
    return df_metadata, exposures, personalities, affinities, adjacency_matrix


def main():
    conf = SimConfig(num_agents=10000, seed=69)
    conf.wealth_dim_idx = DIMENSION_INDICES["Wealth"]
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(conf)
    if conf.enable_evolution:
        evolver = SocietyEvolution(conf, df_meta, exposures, personalities)
        df_meta, exposures, personalities = evolver.evolve()


if __name__ == "__main__":
    main()
