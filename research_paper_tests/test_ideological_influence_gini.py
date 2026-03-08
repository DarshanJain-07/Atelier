import numpy as np
from schema import SimConfig


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


def run_influence_analysis(name, use_power_law=False):
    print(f"\n--- Running Ideological Influence Analysis: {name} ---")
    config = SimConfig(num_agents=10000, seed=42, use_power_law_influence=use_power_law)

    # Generate basic lognormal influence
    influence_scores = np.random.lognormal(
        mean=1.0, sigma=0.5 + config.mutation_temperature, size=config.num_agents
    )

    # Apply Power Law (Pareto) if enabled, similar to the research paper tests
    if use_power_law:
        alpha = 1.16  # standard 80/20 rule pareto
        pareto_multiplier = (np.random.pareto(alpha, config.num_agents) + 1) * 2.0
        influence_scores *= pareto_multiplier

    gini_coeff = gini(influence_scores)

    # Calculate influence distribution
    total_influence = influence_scores.sum()

    # Sort descending to find top influencers
    sorted_influence = np.sort(influence_scores)[::-1]

    top_1_percent = int(config.num_agents * 0.01)
    top_5_percent = int(config.num_agents * 0.05)
    top_20_percent = int(config.num_agents * 0.20)

    top_1_share = sorted_influence[:top_1_percent].sum() / total_influence
    top_5_share = sorted_influence[:top_5_percent].sum() / total_influence
    top_20_share = sorted_influence[:top_20_percent].sum() / total_influence

    print(f"Influence Gini Coefficient: {gini_coeff:.3f}")
    print(f"Top 1% Agents control: {top_1_share*100:.1f}% of total societal influence")
    print(f"Top 5% Agents control: {top_5_share*100:.1f}% of total societal influence")
    print(
        f"Top 20% Agents control: {top_20_share*100:.1f}% of total societal influence"
    )

    return gini_coeff


if __name__ == "__main__":
    run_influence_analysis("Standard Lognormal Influence", use_power_law=False)
    run_influence_analysis("Power-Law (Pareto) Influence", use_power_law=True)
