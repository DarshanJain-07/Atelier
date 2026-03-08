import math
import os

import numpy as np
import pandas as pd
import torch

from schema import (
    DIMENSIONS,
    DIMENSION_INDICES,
    SimConfig,
)
from society_evolution import SocietyEvolution


# ================================
# Mutation Logic
# ================================
def apply_random_mutations(exposures, personalities, temperature, seed):
    if temperature <= 0.0:
        return exposures, personalities

    rng = torch.Generator()
    rng.manual_seed(seed + 999)

    n_agents = exposures.shape[0]
    num_dims = len(DIMENSIONS)

    mutation_prob = temperature
    mutant_mask = torch.rand(n_agents, generator=rng) < mutation_prob
    mutant_indices = torch.where(mutant_mask)[0]

    if len(mutant_indices) == 0:
        return exposures, personalities

    num_changes = math.ceil(3 * temperature)

    for _ in range(num_changes):
        col_indices = torch.randint(0, num_dims, (len(mutant_indices),), generator=rng)
        # Use normal distribution for mutations to preserve bell curve (avoid uniformity/bimodality)
        random_values = torch.clamp(torch.randn(len(mutant_indices), generator=rng) * 0.4, -1.0, 1.0)
        exposures[mutant_indices, col_indices] = random_values

    for _ in range(num_changes):
        col_indices = torch.randint(0, 5, (len(mutant_indices),), generator=rng)
        random_values = torch.rand(len(mutant_indices), generator=rng)
        personalities[mutant_indices, col_indices] = random_values

    return exposures, personalities


# ================================
# HYBRID WEALTH ENGINE
# ================================
def generate_hybrid_wealth(
    role_wealth_bases,
    influence_scores,
    temperature,
    seed,
):
    """
    Multi-regime wealth distribution:
    - Bottom 60%: Exponential
    - Middle 35%: Lognormal
    - Top 5%: Pareto
    Influence percentile drives regime selection.
    """

    np.random.seed(seed + 123)

    n = len(role_wealth_bases)

    # Rank-normalize influence into percentile space
    influence_rank = np.argsort(np.argsort(influence_scores))
    percentiles = influence_rank / (n - 1)

    wealth = np.zeros(n)

    # --- Regime Masks ---
    lower_mask = percentiles < 0.60
    middle_mask = (percentiles >= 0.60) & (percentiles < 0.95)
    upper_mask = percentiles >= 0.95

    # --------------------------
    # 1️⃣ Lower Class - Exponential
    # --------------------------
    scale = 1.0 + temperature
    wealth[lower_mask] = np.random.exponential(scale=scale, size=lower_mask.sum())

    # --------------------------
    # 2️⃣ Middle Class - Lognormal
    # --------------------------
    mu = 0.0 + temperature * 0.5
    sigma = 0.5 + temperature * 0.5
    wealth[middle_mask] = np.random.lognormal(
        mean=mu,
        sigma=sigma,
        size=middle_mask.sum(),
    )

    # --------------------------
    # 3️⃣ Upper Class - Pareto
    # --------------------------
    alpha = 1.5 - (temperature * 0.3)  # heavier tail with temperature
    alpha = max(alpha, 1.1)  # stability floor

    wealth[upper_mask] = (
        np.random.pareto(alpha, upper_mask.sum()) + 1
    ) * 5  # base scaling factor

    # --------------------------
    # Network Compounding Effect
    # --------------------------
    compounding = 1 + (percentiles**2) * (1 + temperature)
    wealth *= compounding

    # Multiply by role structural wealth
    wealth *= role_wealth_bases

    return wealth


# ================================
# Main Generator (Bias-Free)
# ================================
def create_topology(config: SimConfig, exposures: torch.Tensor, personalities: torch.Tensor, influence_scores: np.ndarray):
    """
    Creates a sparse K-Nearest Neighbor adjacency matrix based on influence.
    High influence = larger out-degree (k).
    Edges are weighted by homophily (cosine similarity of traits).
    """
    N = config.num_agents
    
    # 1. Determine out-degree (k) per agent based on influence
    inf_mean = np.mean(influence_scores)
    k_array = np.clip((influence_scores / inf_mean) * config.base_connections, 1, config.max_connections).astype(int)
    
    # 2. Build feature matrix for KNN
    features = torch.cat([exposures, personalities], dim=1)
    
    # L2 normalize features
    features_norm = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)
    
    # Process in batches to save memory
    batch_size = 1000
    indices_list = []
    values_list = []
    
    # Ensure inputs are on the same device
    device = exposures.device
    features_norm = features_norm.to(device)

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch_size_actual = end - i
        batch_features = features_norm[i:end]
        
        # similarity: (batch_size_actual, N)
        sim = torch.mm(batch_features, features_norm.T)
        
        # Add homophily bias. Exponentiate to punish low similarity heavily.
        sim = torch.pow((sim + 1.0) / 2.0, getattr(config, "homophily_strength", 2.0))
        
        # Self-loops must be removed
        # Set self-similarity to -1.0 so they aren't picked by topk
        batch_indices = torch.arange(batch_size_actual, device=device)
        global_indices = batch_indices + i
        sim[batch_indices, global_indices] = -1.0
        
        # Determine k for this batch
        batch_k = k_array[i:end] # numpy array
        max_k = int(np.max(batch_k))
        
        # Get top max_k neighbors
        # (batch_size_actual, max_k)
        top_vals, top_indices = torch.topk(sim, max_k, dim=1)
        
        # Create mask for variable k
        range_tensor = torch.arange(max_k, device=device).unsqueeze(0)
        k_tensor = torch.tensor(batch_k, device=device).unsqueeze(1)
        mask = range_tensor < k_tensor
        
        # Filter valid edges
        valid_src = torch.arange(i, end, device=device).unsqueeze(1).expand(-1, max_k)[mask]
        valid_dst = top_indices[mask]
        valid_val = top_vals[mask]
        
        indices_list.append(torch.stack([valid_src, valid_dst]))
        values_list.append(valid_val)
                
    if not indices_list:
        return None

    indices_tensor = torch.cat(indices_list, dim=1)
    values_tensor = torch.cat(values_list)
    
    sparse_adj = torch.sparse_coo_tensor(indices_tensor, values_tensor, size=(N, N))
    
    # Row normalize so each agent's local center is a proper mean
    dense_sums = torch.sparse.sum(sparse_adj, dim=1).to_dense()
    dense_sums = torch.clamp(dense_sums, min=1e-8)
    
    row_indices = indices_tensor[0]
    normalized_values = values_tensor / dense_sums[row_indices]
    
    normalized_sparse_adj = torch.sparse_coo_tensor(indices_tensor, normalized_values, size=(N, N)).coalesce()
    return normalized_sparse_adj


def generate_society(config: SimConfig):

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    print(f"Generating {config.num_agents} Agents via Bias-Free Continuous Field...")

    num_dims = len(DIMENSIONS)
    wealth_idx = DIMENSION_INDICES["Wealth"]
    num_personalities = 5
    total_dims = num_dims * 2 + num_personalities

    # ---- Continuous random trait field ----
    # Mean 0, Std set by config to fit nicely in [-1, 1] without hard clamping
    traits = torch.randn(config.num_agents, total_dims) * config.initial_trait_std_dev

    exposures = traits[:, :num_dims]
    # Sigmoid to bound personalities between 0 and 1
    personalities = torch.sigmoid(traits[:, num_dims : num_dims + num_personalities])
    raw_affinities = traits[:, num_dims + num_personalities :]

    exposures, personalities = apply_random_mutations(
        exposures, personalities, config.mutation_temperature, config.seed
    )

    # ---- Influence (no role-based anchoring) ----
    influence_scores = np.random.lognormal(
        mean=1.0,
        sigma=0.5 + config.mutation_temperature,
        size=config.num_agents
    )
    
    if getattr(config, "use_power_law_influence", False):
        alpha = 1.16  # standard 80/20 rule pareto
        pareto_multiplier = (np.random.pareto(alpha, config.num_agents) + 1) * 2.0
        influence_scores *= pareto_multiplier

    # To create a realistic correlation between structural influence and susceptibility,
    # we slightly modify personalities of top influencers to be less susceptible 
    # (lower agreeableness, lower openness to external change)
    # and lower influencers to be more standard.
    
    # rank influence to get percentiles
    influence_ranks = np.argsort(np.argsort(influence_scores)) / (config.num_agents - 1)
    
    # Agreeableness is index 3, Openness is index 0
    # Top influencers (rank near 1.0) get a reduction in agreeableness and openness
    stubbornness_modifier = torch.tensor(influence_ranks, dtype=torch.float32).unsqueeze(1) * 0.3
    
    # Reduce agreeableness (index 3) and openness (index 0)
    # Give them a minimum floor so they can still be occasionally swayed (e.g. by board members/peers)
    personalities[:, 0] = torch.clamp(personalities[:, 0] - stubbornness_modifier.squeeze() * 0.4, 0.3, 1.0)
    personalities[:, 3] = torch.clamp(personalities[:, 3] - stubbornness_modifier.squeeze() * 0.4, 0.3, 1.0)


    # ---- Wealth (no role multipliers) ----
    wealth_base = np.ones(config.num_agents)

    wealth_values = generate_hybrid_wealth(
        wealth_base,
        influence_scores,
        config.mutation_temperature,
        config.seed,
    )

    # Normalize wealth to [-1, 1] for the exposures tensor (RDE layer assumes [-1, 1])
    # We use log1p to compress the heavy tail, then min-max scale, then shift.
    log_wealth = np.log1p(wealth_values)
    min_w = np.min(log_wealth)
    max_w = np.max(log_wealth)
    if max_w > min_w:
        wealth_normalized = 2.0 * ((log_wealth - min_w) / (max_w - min_w)) - 1.0
    else:
        wealth_normalized = np.zeros_like(log_wealth)

    exposures[:, wealth_idx] = torch.tensor(wealth_normalized).float()

    # Clamp non-wealth traits to strict bounds (most are already in bounds due to std 0.33)
    non_wealth_mask = torch.ones(num_dims, dtype=torch.bool)
    non_wealth_mask[wealth_idx] = False
    exposures[:, non_wealth_mask] = torch.clamp(exposures[:, non_wealth_mask], -1.0, 1.0)
    personalities = torch.clamp(personalities, 0.0, 1.0)

    # --- Affinity Normalization (Cognitive Bandwidth) ---
    # Apply Bell Curve: mean=0.55, std=0.2, clamped to [0.1, 1.0]
    cognitive_bandwidth = torch.clamp(
        torch.randn(config.num_agents, 1) * 0.2 + 0.55, min=0.1, max=1.0
    )

    # Ensure affinities are positive before normalization (using absolute values or clamping)
    # Using absolute value makes sense for Gaussian noise to capture intensity
    positive_affinities = torch.clamp(torch.abs(raw_affinities), min=0.01)

    # L1 Normalize to sum to 1.0, then scale by bandwidth
    normalized_affinities = positive_affinities / positive_affinities.sum(
        dim=1, keepdim=True
    )
    affinities = normalized_affinities * cognitive_bandwidth

    df_metadata = pd.DataFrame(
        {
            "Agent_ID": range(config.num_agents),
            "Role": ["Agent"] * config.num_agents,
            "Region": ["Global"] * config.num_agents,
            "Influence": np.round(influence_scores, 3),
            "Cognitive_Bandwidth": np.round(cognitive_bandwidth.squeeze().numpy(), 3),
        }
    )

    df_metadata.to_parquet(f"{config.output_dir}/metadata.parquet")
    torch.save(exposures, f"{config.output_dir}/exposures.pt")
    torch.save(personalities, f"{config.output_dir}/personalities.pt")
    torch.save(affinities, f"{config.output_dir}/affinities.pt")

    adjacency_matrix = None
    if getattr(config, "use_network_topology", False):
        print("Generating Network Topology (Sparse KNN Adjacency Matrix)...")
        adjacency_matrix = create_topology(config, exposures, personalities, influence_scores)
        torch.save(adjacency_matrix, f"{config.output_dir}/adjacency.pt")

    print(f"Society Generated in '{config.output_dir}' (Bias-Free)")
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
