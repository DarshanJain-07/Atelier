import math
import os

import numpy as np
import pandas as pd
import torch

from schema import (
    DIMENSIONS,
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
def generate_society(config: SimConfig):

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    print(f"Generating {config.num_agents} Agents via Bias-Free Continuous Field...")

    num_dims = len(DIMENSIONS)
    wealth_idx = DIMENSIONS.index("Wealth")
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

    print(f"Society Generated in '{config.output_dir}' (Bias-Free)")
    return df_metadata, exposures, personalities, affinities


if __name__ == "__main__":
    conf = SimConfig(num_agents=10000, seed=69)
    conf.wealth_dim_idx = DIMENSIONS.index("Wealth")
    df_meta, exposures, personalities, affinities = generate_society(conf)
    if conf.enable_evolution:
        evolver = SocietyEvolution(conf, df_meta, exposures, personalities)
        df_meta, exposures, personalities = evolver.evolve()
