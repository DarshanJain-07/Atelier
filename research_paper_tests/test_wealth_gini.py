import torch
import numpy as np
import pandas as pd
from schema import SimConfig, DIMENSIONS
from generate_society import generate_hybrid_wealth
from society_evolution import SocietyEvolution


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def run_experiment(name, use_hybrid_start=True, generations=300):
    print(f"\n--- Running Experiment: {name} ---")
    config = SimConfig(
        num_agents=5000,
        seed=42,
        enable_evolution=True,
        evolution_generations=generations,
        record_history=False,
    )
    config.wealth_dim_idx = DIMENSIONS.index("Wealth")

    # 1. Generate Base Data (bypassing generate_society to control wealth injection cleanly)
    num_dims = len(DIMENSIONS)
    exposures = torch.zeros(config.num_agents, num_dims)
    personalities = torch.rand(config.num_agents, 5)
    influence_scores = np.random.lognormal(
        mean=1.0, sigma=0.5 + config.mutation_temperature, size=config.num_agents
    )

    df_metadata = pd.DataFrame(
        {
            "Agent_ID": range(config.num_agents),
            "Role": ["Agent"] * config.num_agents,
            "Region": ["Global"] * config.num_agents,
            "Influence": np.round(influence_scores, 3),
            "Cognitive_Bandwidth": np.random.normal(0.55, 0.2, config.num_agents),
        }
    )

    # 2. Inject Initial Wealth
    if use_hybrid_start:
        wealth = generate_hybrid_wealth(
            np.ones(config.num_agents),
            influence_scores,
            config.mutation_temperature,
            config.seed,
        )
    else:
        # Uniform start: everyone gets a baseline wealth
        wealth = np.ones(config.num_agents) * 10.0

    exposures[:, config.wealth_dim_idx] = torch.tensor(wealth).float()

    # Calculate Initial Stats
    initial_wealth = exposures[:, config.wealth_dim_idx].numpy()
    initial_gini = gini(initial_wealth)
    top_5_percentile = np.percentile(initial_wealth, 95)
    top_5_share = (
        initial_wealth[initial_wealth >= top_5_percentile].sum() / initial_wealth.sum()
    )

    print(
        f"Gen 0   | Gini: {initial_gini:.3f} | Top 5% hold {top_5_share*100:.1f}% of wealth"
    )

    # 3. Evolve
    evolver = SocietyEvolution(config, df_metadata, exposures, personalities)
    # We will step through manually to print checkpoints
    for gen in range(1, generations + 1):
        evolver.apply_inheritance()
        evolver.apply_reinvestment()
        evolver.apply_economic_shocks(gen)
        evolver.apply_mobility()
        evolver.apply_mobility()

        evolver.exposures[:, evolver.wealth_idx] = torch.clamp(
            evolver.exposures[:, evolver.wealth_idx], min=0.0, max=1e6
        )

        if gen in [10, 50, 100, 300]:
            current_wealth = evolver.exposures[:, evolver.wealth_idx].numpy()
            current_gini = gini(current_wealth)
            top_5_percentile = np.percentile(current_wealth, 95)
            top_5_share = (
                current_wealth[current_wealth >= top_5_percentile].sum()
                / current_wealth.sum()
            )
            print(
                f"Gen {gen:<3} | Gini: {current_gini:.3f} | Top 5% hold {top_5_share*100:.1f}% of wealth"
            )

    return current_gini


if __name__ == "__main__":
    run_experiment(
        "Current Hybrid Power-Law Start", use_hybrid_start=True, generations=300
    )
    run_experiment("Uniform Base Wealth Start", use_hybrid_start=False, generations=300)
