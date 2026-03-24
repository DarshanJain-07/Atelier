import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from generate_society import generate_society
from schema import DIMENSION_INDICES, SimConfig


def to_numpy(x):
    """Safely convert tensors or arrays to numpy."""
    if hasattr(x, "detach"):  # PyTorch tensor
        return x.detach().cpu().numpy()
    elif hasattr(x, "numpy"):  # already numpy-like
        return x.numpy()
    return np.array(x)


def visualize(seed=42):
    # 1. Generate Society
    config = SimConfig(num_agents=5000, seed=seed)
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)

    # 2. Extract Data
    trait_names = [
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism",
    ]

    pers_data = to_numpy(personalities)
    exposures = to_numpy(exposures)

    # Wealth and Influence
    wealth_idx = DIMENSION_INDICES["Wealth"]
    wealth = exposures[:, wealth_idx]

    influence = df_meta["Influence"].values

    # 🔒 Fix log-scale issues
    influence = np.clip(influence, 1e-6, None)

    # 3. Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 15))
    fig.suptitle(
        f"Agent Traits and Wealth vs Influence Analysis (Seed: {seed})",
        fontsize=22
    )

    axes = axes.flatten()

    # Personality Trait Subplots
    for i, name in enumerate(trait_names):
        sns.histplot(
            pers_data[:, i],
            kde=True,
            ax=axes[i],
            color=sns.color_palette("viridis")[i],
        )
        axes[i].set_title(f"{name} Distribution")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")

    # Wealth vs Influence Subplot (last plot)
    sns.scatterplot(
        x=wealth,
        y=influence,
        ax=axes[-1],
        alpha=0.5,
        color="red"
    )
    axes[-1].set_title("Wealth vs Influence")
    axes[-1].set_xlabel("Normalized Wealth")
    axes[-1].set_ylabel("Influence Score (Log-Scale)")
    axes[-1].set_yscale("log")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot
    output_filename = f"traits_wealth_analysis_seed_{seed}.png"
    plt.savefig(output_filename, dpi=300)
    plt.close(fig)

    print(f"Visualization saved to {output_filename}")


if __name__ == "__main__":
    visualize(seed=42)