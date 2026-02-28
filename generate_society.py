import math
import os

import numpy as np
import pandas as pd
import torch

from schema import (
    ARCHETYPES,
    DIMENSIONS,
    REGION_ROLE_PROBS,
    REGION_WEIGHTS,
    REGIONS,
    ROLE_STARTING_STATS,
    ROLE_TIER_MAP,
    ROLES,
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
        random_values = (torch.rand(len(mutant_indices), generator=rng) * 2) - 1.0
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
# Main Generator
# ================================
def generate_society(config: SimConfig):

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    print(f"Generating {config.num_agents} Agents via Density Field...")

    num_dims = len(DIMENSIONS)
    wealth_idx = DIMENSIONS.index("Wealth")

    # --- Density Construction ---
    all_role_means = []
    for role in ROLES:
        dna = torch.cat([ARCHETYPES[role]["exp"], ARCHETYPES[role]["big5"]])
        all_role_means.append(dna)

    means_tensor = torch.stack(all_role_means)

    regions_arr = np.random.choice(REGIONS, size=config.num_agents, p=REGION_WEIGHTS)
    final_traits = torch.zeros(config.num_agents, 17)

    for region in REGIONS:
        mask = torch.from_numpy(regions_arr == region)
        count = int(mask.sum().item())
        if count == 0:
            continue

        role_weights = torch.tensor(REGION_ROLE_PROBS[region])
        component_indices = torch.multinomial(role_weights, count, replacement=True)

        std = 0.15 + (config.mutation_temperature * 0.1)
        batch_means = means_tensor[component_indices]
        noise = torch.randn(count, 17) * std

        final_traits[mask] = batch_means + noise

    exposures = final_traits[:, :num_dims]
    personalities = final_traits[:, num_dims:]

    exposures, personalities = apply_random_mutations(
        exposures, personalities, config.mutation_temperature, config.seed
    )

    # --- Classification ---
    dist_to_archetypes = torch.cdist(final_traits, means_tensor)
    closest_role_indices = torch.argmin(dist_to_archetypes, dim=1)
    roles_arr = np.array([ROLES[idx] for idx in closest_role_indices])
    tiers_arr = np.array([ROLE_TIER_MAP[r] for r in roles_arr])

    role_wealth_bases = np.array([ROLE_STARTING_STATS[r][0] for r in roles_arr])
    role_influence_bases = np.array([ROLE_STARTING_STATS[r][1] for r in roles_arr])

    # ---------------------------
    # Influence Generation
    # ---------------------------
    influence_scores = role_influence_bases * (
        1 + np.random.lognormal(0, 0.5 + config.mutation_temperature, config.num_agents)
    )

    # ---------------------------
    # Hybrid Wealth Generation
    # ---------------------------
    wealth_values = generate_hybrid_wealth(
        role_wealth_bases,
        influence_scores,
        config.mutation_temperature,
        config.seed,
    )

    exposures[:, wealth_idx] = torch.tensor(wealth_values).float()

    # Clamp non-wealth traits only
    non_wealth_mask = torch.ones(num_dims, dtype=torch.bool)
    non_wealth_mask[wealth_idx] = False
    exposures[:, non_wealth_mask] = torch.clamp(
        exposures[:, non_wealth_mask], -1.0, 1.0
    )
    personalities = torch.clamp(personalities, 0.0, 1.0)

    df_metadata = pd.DataFrame(
        {
            "Agent_ID": range(config.num_agents),
            "Role": roles_arr,
            "Region": regions_arr,
            "Tier": tiers_arr,
            "Influence": np.round(influence_scores, 3),
        }
    )

    df_metadata.to_parquet(f"{config.output_dir}/metadata.parquet")
    torch.save(exposures, f"{config.output_dir}/exposures.pt")
    torch.save(personalities, f"{config.output_dir}/personalities.pt")

    print(f"Society Generated in '{config.output_dir}' (Hybrid Wealth + Density)")
    return df_metadata, exposures, personalities


if __name__ == "__main__":
    conf = SimConfig(num_agents=10000, seed=69)
    conf.wealth_dim_idx = DIMENSIONS.index("Wealth")
    df_meta, exposures, personalities = generate_society(conf)
    if conf.enable_evolution:
        evolver = SocietyEvolution(conf, df_meta, exposures, personalities)
        df_meta, exposures, personalities = evolver.evolve()
