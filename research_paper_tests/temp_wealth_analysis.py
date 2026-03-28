import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import spearmanr

from generate_society import generate_society
from research_paper_tests.config_schema import get_test_scenario


def visualize(seed=42):
    print(f"--- VERIFYING NETWORK SYNERGY WEALTH [Seed: {seed}] ---")
    scenario = get_test_scenario("temp_wealth_analysis")
    settings = scenario.settings()
    
    # 1. Generate Society
    config = scenario.sim_config(seed=seed)
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)

    # 2. Extract Data
    raw_wealth = df_meta["Raw_Wealth"].values
    influence = df_meta["Influence"].values
    
    # Calculate in-degrees for correlation check
    adj_coalesced = adjacency_matrix.coalesce()
    indices = adj_coalesced.indices()
    in_degrees = torch.bincount(indices[1], minlength=len(df_meta)).float().numpy()

    # 3. Calculate Correlations
    corr_inf, _ = spearmanr(raw_wealth, influence)
    corr_deg, _ = spearmanr(raw_wealth, in_degrees)
    
    print("\n[CORRELATION ANALYSIS]")
    print(f"Wealth vs Influence: {corr_inf:.4f}")
    print(f"Wealth vs In-Degree (Social Capital): {corr_deg:.4f}")

    # 4. Distribution Stats
    print("\n[WEALTH STATS]")
    print(f"Mean Wealth: {np.mean(raw_wealth):.2f}")
    print(f"Median Wealth: {np.median(raw_wealth):.2f}")
    print(f"75th Percentile: {np.percentile(raw_wealth, 75):.2f}")
    print(f"95th Percentile: {np.percentile(raw_wealth, 95):.2f}")
    print(f"Max Wealth: {np.max(raw_wealth):.2f}")
    print(f"Min Wealth: {np.min(raw_wealth):.2f}")
    
    wealth_over_threshold = np.sum(raw_wealth > settings["wealth_threshold"])
    print(
        f"Agents with Wealth > {settings['wealth_threshold']:,}: "
        f"{wealth_over_threshold} ({wealth_over_threshold / len(raw_wealth) * 100:.2f}%)"
    )

    # 5. Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot 1: Wealth vs Influence (Scatter)
    sns.scatterplot(
        x=raw_wealth,
        y=influence,
        alpha=settings["scatter_alpha"],
        color="teal",
        ax=axes[0],
        edgecolor=None
    )
    axes[0].set_title(f"Wealth vs Influence\nSynergy Model", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Raw Wealth", fontsize=12)
    axes[0].set_ylabel("Influence Score", fontsize=12)

    # Plot 2: Wealth Density Distribution
    sns.histplot(
        raw_wealth,
        kde=True,
        ax=axes[1],
        color="purple",
        bins=settings["hist_bins"],
    )
    axes[1].set_title("Wealth Distribution (Network Clustered)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Raw Wealth", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_xlim(0, np.percentile(raw_wealth, settings["x_axis_percentile"])) # Focus on the bulk

    # Plot 3: Box Plot (Outliers)
    sns.boxplot(x=raw_wealth, ax=axes[2], color="gold", fliersize=2)
    axes[2].set_title("Wealth Outlier Analysis", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Raw Wealth", fontsize=12)
    # ax2.set_xscale('log') # Optional: log scale to see billionaires better

    plt.tight_layout()
    output_filename = settings["output_template"].format(seed=seed)
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSuccess: Verification visualization saved to {output_filename}")


if __name__ == "__main__":
    visualize(seed=42)
