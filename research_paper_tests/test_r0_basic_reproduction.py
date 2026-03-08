import numpy as np
import torch
import sys
import os
from scipy.stats import pearsonr

# Add parent directory to path to import schema, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import SimConfig
from generate_society import generate_society
from cognitive_engine import CognitiveEngine


def test_r0_basic_reproduction():
    print("--- Running R0 (Basic Reproduction Number) Analysis ---")
    config = SimConfig(num_agents=2000, seed=42)

    print("\n[ Generating Society ]")
    df_meta, exposures, personalities, affinities, _ = generate_society(config)

    cog_engine = CognitiveEngine(config)

    # Calculate societal mean ideology
    societal_mean = exposures.mean(dim=0)

    # Select random agents to act as seeds
    num_seeds = 200
    np.random.seed(42)
    seed_indices = np.random.choice(config.num_agents, num_seeds, replace=False)

    # We will use a fixed threshold that signifies a "highly engaged" agent
    # (they are influenced enough to propagate the thought)
    threshold = 0.5

    r0_values = []
    extremity_scores = []

    print(f"\n[ Simulating Thought Propagation for {num_seeds} seeds ]")
    for idx in seed_indices:
        # The "thought" shared by the agent is their ideological vector
        thought_vector = exposures[idx]

        # Calculate how extreme the thought is (distance from societal mean)
        extremity = torch.norm(thought_vector - societal_mean).item()
        extremity_scores.append(extremity)

        # Simulate how the rest of society reacts to this thought
        _, _, engagement_scores, _ = cog_engine.run(
            world_tensor_raw=thought_vector.unsqueeze(0),
            urgency=0.5,
            is_personal=False,
            exposures=exposures,
            personalities=personalities,
            agent_affinities=affinities,
        )

        # Exclude the seed agent themselves
        mask = torch.ones(config.num_agents, dtype=torch.bool)
        mask[idx] = False

        # Count influenced agents
        influenced_count = (engagement_scores[mask] > threshold).sum().item()
        r0_values.append(influenced_count)

    print("\n[ Results ]")
    print(
        "Definition: R0 is the average number of secondary agents highly engaged (score > 0.5) by a single agent's thought."
    )
    print(
        "If Average R0 > 1, the network is fundamentally susceptible to viral cascades.\n"
    )

    mean_r0 = np.mean(r0_values)
    std_r0 = np.std(r0_values)
    max_r0 = np.max(r0_values)
    super_spreaders = sum(1 for r in r0_values if r >= 5)

    cascade = "YES" if mean_r0 > 1.0 else "NO"

    print(f"Average R0:             {mean_r0:.3f} (± {std_r0:.3f})")
    print(f"Max R0 (Most Viral):    {max_r0}")
    print(
        f"Super-spreaders (R0≥5): {super_spreaders} out of {num_seeds} seeds ({super_spreaders/num_seeds*100:.1f}%)"
    )
    print(f"Viral Cascade Potential: {cascade}")

    # Check for systemic bias (Do extreme thoughts spread faster?)
    corr, p_val = pearsonr(extremity_scores, r0_values)
    print("\n[ Bias Check: Extremity vs Virality ]")
    print(
        f"Correlation between Thought Extremity and R0: {corr:.3f} (p-value: {p_val:.4f})"
    )

    if corr > 0.3 and p_val < 0.05:
        print(
            "-> BIAS DETECTED: The society favors and amplifies extreme fringe thoughts."
        )
    elif corr < -0.3 and p_val < 0.05:
        print(
            "-> BIAS DETECTED: The society heavily suppresses extreme thoughts, favoring only moderate views."
        )
    else:
        print(
            "-> NO SIGNIFICANT BIAS: Virality is not purely driven by how extreme the thought is."
        )


if __name__ == "__main__":
    test_r0_basic_reproduction()
