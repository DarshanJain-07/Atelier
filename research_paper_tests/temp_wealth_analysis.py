import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import spearmanr

from generate_society import generate_society
from research_paper_tests.config_schema import get_test_scenario
from research_paper_tests.plotting_utils import (
    PAPER_PALETTE,
    apply_paper_style,
    save_paper_figure,
)

apply_paper_style()


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
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot 1: Wealth vs Influence (Scatter)
    sns.scatterplot(
        x=raw_wealth,
        y=influence,
        alpha=settings["scatter_alpha"],
        color=PAPER_PALETTE["primary"],
        ax=axes[0],
        edgecolor=None
    )
    axes[0].set_title("Wealth vs Influence\nSynergy Model")
    axes[0].set_xlabel("Raw Wealth")
    axes[0].set_ylabel("Influence Score")

    # Plot 2: Wealth Density Distribution
    sns.histplot(
        raw_wealth,
        kde=True,
        ax=axes[1],
        color=PAPER_PALETTE["tertiary"],
        bins=settings["hist_bins"],
    )
    axes[1].set_title("Wealth Distribution (Network Clustered)")
    axes[1].set_xlabel("Raw Wealth")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlim(0, np.percentile(raw_wealth, settings["x_axis_percentile"])) # Focus on the bulk

    # Plot 3: Box Plot (Outliers)
    sns.boxplot(x=raw_wealth, ax=axes[2], color=PAPER_PALETTE["accent"], fliersize=2)
    axes[2].set_title("Wealth Outlier Analysis")
    axes[2].set_xlabel("Raw Wealth")
    # ax2.set_xscale('log') # Optional: log scale to see billionaires better

    plt.tight_layout()
    output_filename = settings["output_template"].format(seed=seed)
    save_paper_figure(fig, output_filename)
    plt.close()

    print(f"\nSuccess: Verification visualization saved to {output_filename}")


if __name__ == "__main__":
    visualize(seed=42)
