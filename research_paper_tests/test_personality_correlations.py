import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

# Add parent directory to path to import schema
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generate_society import generate_society
from schema import PERSONALITY_CORRELATIONS, SimConfig


def test_personality_correlations():
    # 1. Setup Config
    config = SimConfig(
        num_agents=35000,  # High count for statistical significance
        seed=42,
        output_dir="test_output_temp",
        mutation_temperature=0,
    )

    # 2. Generate Society (This runs the Cholesky logic)
    print("Generating society...")
    _, _, personalities, _, _ = generate_society(config)

    # 3. Calculate Observed Correlations
    # personalities shape: (N, 5) -> O, C, E, A, N
    trait_names = [
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism",
    ]

    # Convert to Pandas for easy correlation
    df = pd.DataFrame(personalities.numpy(), columns=trait_names)
    observed_corr = df.corr()

    # 4. Get Target Correlations
    target_corr = pd.DataFrame(
        PERSONALITY_CORRELATIONS.numpy(), index=trait_names, columns=trait_names
    )

    print("\n--- Target Correlation Matrix (Schema) ---")
    print(target_corr.round(3))

    print("\n--- Observed Correlation Matrix (Simulation) ---")
    print(observed_corr.round(3))

    # 5. Calculate Error (RMSE)
    diff = observed_corr - target_corr
    rmse = (diff**2).mean().mean() ** 0.5
    print(f"\nRMSE between Target and Observed: {rmse:.4f}")

    # 6. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(target_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("Target Correlations (Schema)")

    sns.heatmap(observed_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("Observed Correlations (Generated)")

    sns.heatmap(diff, annot=True, cmap="coolwarm", vmin=-0.2, vmax=0.2, ax=axes[2])
    axes[2].set_title("Difference (Observed - Target)")

    output_path = "research_paper_tests/personality_correlation_check.png"
    plt.savefig(output_path)
    print(f"\nVisualization saved to: {output_path}")

    # Cleanup
    import shutil

    if os.path.exists("test_output_temp"):
        shutil.rmtree("test_output_temp")


if __name__ == "__main__":
    test_personality_correlations()
