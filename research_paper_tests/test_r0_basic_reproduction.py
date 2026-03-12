import numpy as np
import torch
import sys
import os
import scipy.stats as st
from scipy.stats import pearsonr

# Add parent directory to path to import schema, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import SimConfig
from generate_society import generate_society
from cognitive_engine import CognitiveEngine


def get_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def test_r0_basic_reproduction():
    print("--- Running R0 (Basic Reproduction Number) Analysis (Monte Carlo) ---")
    
    num_runs = 10
    num_seeds_per_run = 50
    threshold = 0.5
    
    all_r0_means = []
    all_corrs = []
    
    for run in range(num_runs):
        seed = 42 + run
        config = SimConfig(num_agents=2000, seed=seed)

        df_meta, exposures, personalities, affinities, _ = generate_society(config)
        cog_engine = CognitiveEngine(config)

        societal_mean = exposures.mean(dim=0)
        np.random.seed(seed)
        seed_indices = np.random.choice(config.num_agents, num_seeds_per_run, replace=False)

        r0_values = []
        extremity_scores = []

        for idx in seed_indices:
            thought_vector = exposures[idx]
            extremity = torch.norm(thought_vector - societal_mean).item()
            extremity_scores.append(extremity)

            _, _, engagement_scores, _ = cog_engine.run(
                world_tensor_raw=thought_vector.unsqueeze(0),
                urgency=0.5,
                is_personal=False,
                exposures=exposures,
                personalities=personalities,
                agent_affinities=affinities,
            )

            mask = torch.ones(config.num_agents, dtype=torch.bool)
            mask[idx] = False

            influenced_count = (engagement_scores[mask] > threshold).sum().item()
            r0_values.append(influenced_count)
            
        all_r0_means.append(np.mean(r0_values))
        corr, p_val = pearsonr(extremity_scores, r0_values)
        if not np.isnan(corr):
            all_corrs.append(corr)

    print("\n[ Results ]")
    print("Definition: R0 is the average number of secondary agents highly engaged (score > 0.5) by a single agent's thought.")
    print("If Average R0 > 1, the network is fundamentally susceptible to viral cascades.\n")

    m_r0, h_r0 = get_confidence_interval(all_r0_means)
    m_corr, h_corr = get_confidence_interval(all_corrs)

    cascade = "YES" if m_r0 > 1.0 else "NO"

    print(f"Average R0 (Over {num_runs} societies): {m_r0:.3f} ± {h_r0:.3f} (95% CI)")
    print(f"Viral Cascade Potential: {cascade}")

    print("\n[ Bias Check: Extremity vs Virality ]")
    print(f"Mean Correlation between Thought Extremity and R0: {m_corr:.3f} ± {h_corr:.3f} (95% CI)")

    if m_corr > 0.3:
        print("-> BIAS DETECTED: The society favors and amplifies extreme fringe thoughts.")
    elif m_corr < -0.3:
        print("-> BIAS DETECTED: The society heavily suppresses extreme thoughts, favoring only moderate views.")
    else:
        print("-> NO SIGNIFICANT BIAS: Virality is not purely driven by how extreme the thought is.")


if __name__ == "__main__":
    test_r0_basic_reproduction()
