import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import pearsonr

# Add parent directory to path to import schema, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_society import generate_society
from schema import SimConfig


def test_signal_distortion():
    print("--- Testing Big-Five Driven Stochastic Signal Distortion ---")

    # 1. Configuration - ensure distortion is ON
    config = SimConfig(
        num_agents=2000,
        use_signal_distortion=True,
        distortion_max_noise=0.8,
        distortion_neurotic_gain=1.5,
    )

    print("Generating Society...")
    _, _, personalities, _, _ = generate_society(config)

    # Extract Neuroticism (Big-Five Index 4: O, C, E, A, N)
    neuroticism_scores = personalities[:, 4].numpy()

    # 2. The Objective Event: A completely neutral / very mildly negative event
    # We will set 'Physical Safety' (idx 1) to -0.1
    world_tensor_raw = torch.zeros(1, 12)
    world_tensor_raw[0, 1] = -0.4

    print(
        f"\nBroadcasting Objective World Tensor: Physical Safety = {world_tensor_raw[0, 1].item()}"
    )

    # 3. Intercept the Signal BEFORE Attention via the Cognitive Engine's internal method
    # Note: We are simulating the distortion layer directly to measure exactly what was perceived

    expanded_world = world_tensor_raw.expand(config.num_agents, -1)

    # Apply standard distortion logic from CognitiveEngine
    if getattr(config, "use_signal_distortion", True):
        max_noise = getattr(config, "distortion_max_noise", 0.4)
        neurotic_gain = getattr(config, "distortion_neurotic_gain", 0.6)

        # Base noise from a beta distribution (skewed towards lower values)
        noise = (
            torch.distributions.Beta(2, 5)
            .sample((config.num_agents, 12))
            .to(expanded_world.device)
        )

        # Scale noise by Neuroticism (trait index 4)
        neuroticism = personalities[:, 4].unsqueeze(1)
        scaled_noise = noise * max_noise * (1.0 + neurotic_gain * neuroticism)

        # For negative signals (threats), noise makes them MORE negative (exaggeration)
        sign = torch.sign(expanded_world)
        sign = torch.where(sign == 0, torch.tensor(1.0, device=sign.device), sign)

        perceived_world_tensor = expanded_world + (scaled_noise * sign)
        perceived_world_tensor = torch.clamp(perceived_world_tensor, -1.0, 1.0)
    else:
        perceived_world_tensor = expanded_world

    # Extract the perceived Physical Safety value for all agents
    perceived_safety = perceived_world_tensor[:, 1].numpy()

    # 4. Calculate the "Distortion Error" (Perceived - Objective)
    # Since objective was -0.1, if they perceive -0.5, the error (exaggeration) is magnitude 0.4
    objective_val = world_tensor_raw[0, 1].item()
    distortion_magnitude = np.abs(perceived_safety - objective_val)

    # 5. Statistical Correlation
    corr, p_value = pearsonr(neuroticism_scores, distortion_magnitude)
    print("\nStatistical Results:")
    print(
        f"Pearson Correlation (Neuroticism vs. Threat Exaggeration): r = {corr:.4f}, p-value = {p_value:.4e}"
    )
    print(
        f"Average Threat Exaggeration for Low Neuroticism (bottom 20%): {np.mean(distortion_magnitude[neuroticism_scores < np.percentile(neuroticism_scores, 20)]):.4f}"
    )
    print(
        f"Average Threat Exaggeration for High Neuroticism (top 20%): {np.mean(distortion_magnitude[neuroticism_scores > np.percentile(neuroticism_scores, 80)]):.4f}"
    )

    # 6. Visualization
    plt.figure(figsize=(10, 6))

    # Create a scatter plot with a regression line
    sns.regplot(
        x=neuroticism_scores,
        y=distortion_magnitude,
        scatter_kws={"alpha": 0.3, "s": 10},
        line_kws={"color": "red", "linewidth": 2},
    )

    plt.title(
        "Neuroticism vs. Signal Distortion (Threat Exaggeration)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Agent Neuroticism Trait [0.0 to 1.0]", fontsize=12)
    plt.ylabel("Magnitude of Threat Exaggeration", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Add text box with stats
    textstr = f"Correlation: r = {corr:.3f}\nObjective Stimulus = -0.4"
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    plt.gca().text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )

    output_path = os.path.join(
        os.path.dirname(__file__), "signal_distortion_neuroticism.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved visualization to: {output_path}")


if __name__ == "__main__":
    test_signal_distortion()
