import numpy as np
from schema import SimConfig, DIMENSIONS
from generate_society import generate_society
from society_evolution import SocietyEvolution


def calculate_bimodality_coefficient(data):
    """
    Calculate the Bimodality Coefficient (BC) of a 1D array using pure numpy.
    BC = (skewness^2 + 1) / kurtosis
    Where kurtosis is the standard Pearson's kurtosis (normal = 3.0).
    BC > 0.555 typically indicates bimodality (polarization).
    """
    data = np.asarray(data).flatten()

    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        return 0.0

    # Sample Skewness
    skew = np.mean(((data - mean) / std) ** 3)

    # Pearson's Kurtosis (Normal distribution = 3.0)
    kurtosis = np.mean(((data - mean) / std) ** 4)

    if kurtosis == 0:
        return 0.0

    # Empirical Bimodality Coefficient
    bc = (skew**2 + 1) / kurtosis
    return bc


def run_polarization_test():
    print("--- Running Bimodality Coefficient (Polarization) Analysis ---")
    config = SimConfig(
        num_agents=5000,
        seed=3782,
        enable_evolution=True,
        evolution_generations=200,
        record_history=True,
    )
    # Ensure config has wealth index
    config.wealth_dim_idx = DIMENSIONS.index("Wealth")

    # Suppress generator prints for clean output
    print("\nGenerating Initial Society...")
    df_meta, exposures, personalities, affinities, _ = generate_society(config)

    # Track a few key dimensions
    dims_to_track = ["Fairness", "Innovation", "In_Group", "Wealth"]

    print("\n[ Initial Distribution Analysis ]")
    print("Note: BC > 0.555 indicates Polarization (Bimodality)")
    initial_bcs = {}
    for dim in dims_to_track:
        dim_idx = DIMENSIONS.index(dim)
        data = exposures[:, dim_idx].numpy()
        bc = calculate_bimodality_coefficient(data)
        initial_bcs[dim] = bc
        print(f"  {dim:12s} | Initial BC: {bc:.3f}")

    # Math verification with perfect examples
    polarized_data = np.concatenate(
        [np.random.normal(-0.8, 0.1, 2500), np.random.normal(0.8, 0.1, 2500)]
    )
    forced_bc = calculate_bimodality_coefficient(polarized_data)
    print("\n[ Math Verification ]")
    print(f"  Perfect 50/50 Split (Echo Chambers) BC: {forced_bc:.3f} (Should be ~1.0)")

    normal_data = np.random.normal(0, 1, 5000)
    normal_bc = calculate_bimodality_coefficient(normal_data)
    print(f"  Perfect Consensus (Bell Curve) BC: {normal_bc:.3f} (Should be ~0.33)")

    # Evolve society
    print(f"\n[ Evolving Society ({config.evolution_generations} Generations) ]")
    evolver = SocietyEvolution(config, df_meta, exposures, personalities)
    df_meta, exposures_final, personalities_final = evolver.evolve()

    print("\n[ Final Distribution Analysis ]")
    for dim in dims_to_track:
        dim_idx = DIMENSIONS.index(dim)
        data = exposures_final[:, dim_idx].numpy()
        bc = calculate_bimodality_coefficient(data)

        diff = bc - initial_bcs[dim]
        trend = "INCREASING (Polarizing)" if diff > 0 else "DECREASING (Consensus)"

        print(
            f"  {dim:12s} | Final BC: {bc:.3f} | Change: {diff:+.3f} | Trend: {trend}"
        )
        if bc > 0.555:
            print(f"    -> WARNING: The society is structurally polarized on {dim}.")


if __name__ == "__main__":
    run_polarization_test()
