import numpy as np
import torch

from generate_society import generate_society
from schema import DIMENSIONS, SimConfig
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
        evolution_generations=100,
        record_history=True,
    )
    # Ensure config has wealth index
    config.wealth_dim_idx = DIMENSIONS.index("Wealth")

    # Suppress generator prints for clean output
    print("\nGenerating Initial Society...")
    df_meta, exposures, personalities, affinities, _ = generate_society(config)

    # Track a few key dimensions
    dims_to_track = ["Fairness", "Innovation", "In_Group", "Wealth"]
    dim_idx_fairness = DIMENSIONS.index("Fairness")

    # Storage for joyplot
    history_fairness = []

    print("\n[ Initial Distribution Analysis ]")
    print("Note: BC > 0.555 indicates Polarization (Bimodality)")
    initial_bcs = {}
    for dim in dims_to_track:
        dim_idx = DIMENSIONS.index(dim)
        data = exposures[:, dim_idx].numpy()
        bc = calculate_bimodality_coefficient(data)
        initial_bcs[dim] = bc
        print(f"  {dim:12s} | Initial BC: {bc:.3f}")

    history_fairness.append(exposures[:, dim_idx_fairness].clone().numpy())

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

    # Evolve society manually to capture history
    print(f"\n[ Evolving Society ({config.evolution_generations} Generations) ]")
    evolver = SocietyEvolution(config, df_meta, exposures, personalities)

    for gen in range(1, config.evolution_generations + 1):
        evolver.apply_inheritance()
        evolver.apply_reinvestment()
        evolver.apply_economic_shocks(gen)
        evolver.apply_mobility()
        evolver.apply_ideological_drift()

        if getattr(config, "use_dynamic_classes", False):
            evolver.reassign_classes()

        evolver.exposures[:, evolver.wealth_idx] = torch.clamp(
            evolver.exposures[:, evolver.wealth_idx], min=0.0, max=1e6
        )

        # Capture state every 10 generations
        if gen % 10 == 0:
            history_fairness.append(
                evolver.exposures[:, dim_idx_fairness].clone().numpy()
            )

    df_meta, exposures_final, _ = (
        evolver.metadata,
        evolver.exposures,
        evolver.personalities,
    )

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

    # Generate Ridge Plot (Joyplot) using matplotlib & seaborn
    try:
        import os

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(len(history_fairness), 1, figsize=(8, 10), sharex=True)
        fig.subplots_adjust(hspace=-0.5)

        for i, (ax, data) in enumerate(zip(axes, history_fairness)):
            sns.kdeplot(data, ax=ax, fill=True, alpha=0.6, linewidth=1.5)
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            gen_label = f"Gen {i * 10}" if i > 0 else "Gen 0"
            ax.text(-1.1, 0, gen_label, fontweight="bold", fontsize=10, ha="right")

        axes[-1].set_xlabel("Fairness Ideology Score")
        fig.suptitle(
            "Evolution of 'Fairness' Ideology over 100 Generations", y=0.95, fontsize=14
        )

        output_path = os.path.join(
            os.path.dirname(__file__), "bimodality_ridge_plot.png"
        )
        plt.savefig(output_path, bbox_inches="tight")
        print(f"\n[!] Saved Ridge Plot to: {output_path}")

    except ImportError:
        print("\n[!] matplotlib/seaborn not installed. Skipping plot generation.")


if __name__ == "__main__":
    run_polarization_test()
