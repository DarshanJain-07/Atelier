import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent directory to path to import schema, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_engine import CognitiveEngine
from generate_society import generate_society
from schema import DIMENSION_INDICES, SimConfig


def test_cognitive_gate():
    print("--- Testing Confirmation Bias / Selective Exposure Gate ---")

    # 1. Configuration
    config = SimConfig(
        num_agents=2000, 
        use_signal_distortion=False, # Disable noise to perfectly measure the gate
    )
    
    # Enable selective exposure explicitly just in case
    setattr(config, "use_selective_exposure", True)
    setattr(config, "selective_exposure_base_tolerance", -0.3)
    setattr(config, "selective_exposure_openness_factor", 0.4)

    print("Generating Base Society...")
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)

    # 2. Artificially create two polarized groups
    # First 1000 agents: "Extreme Traditionalists" (Low Openness, Conservative Worldview)
    # Last 1000 agents: "Extreme Progressives" (High Openness, Progressive Worldview)

    half = config.num_agents // 2

    # Override Personalities:
    # Big Five Index 0 is Openness
    personalities[:half, 0] = 0.1  # Low Openness
    personalities[half:, 0] = 0.9  # High Openness

    # Override Exposures (Worldview vectors)
    # Dims: Innovation(6), Fairness(4), Sanctity(8), In_Group(5)
    idx_innov = DIMENSION_INDICES["Innovation"]
    idx_fair = DIMENSION_INDICES["Fairness"]
    idx_sanc = DIMENSION_INDICES["Sanctity"]
    idx_ingroup = DIMENSION_INDICES["In_Group"]

    # Clear exposures
    exposures.zero_()

    # Traditionalists: High Sanctity, High In-Group, Low Innovation, Low Fairness (for this specific toy example)
    exposures[:half, idx_innov] = -1.0
    exposures[:half, idx_fair] = -1.0
    exposures[:half, idx_sanc] = 1.0
    exposures[:half, idx_ingroup] = 1.0

    # Progressives: High Innovation, High Fairness, Low Sanctity, Low In-Group
    exposures[half:, idx_innov] = 1.0
    exposures[half:, idx_fair] = 1.0
    exposures[half:, idx_sanc] = -1.0
    exposures[half:, idx_ingroup] = -1.0

    # 3. Create a Highly Progressive Event
    # This event fundamentally contradicts the Traditionalists' worldview
    world_tensor_raw = torch.zeros(1, 12)
    world_tensor_raw[0, idx_innov] = 0.8
    world_tensor_raw[0, idx_fair] = 0.7
    world_tensor_raw[0, idx_sanc] = -0.9
    world_tensor_raw[0, idx_ingroup] = -0.5

    urgency = 0.2  # Low urgency so stress bias doesn't interfere
    is_personal = False

    print("\nBroadcasting Highly Progressive Event...")
    
    agent_memory = torch.zeros_like(exposures)
    cog_engine = CognitiveEngine(config)
    
    ctx, att, eng, agent_memory = cog_engine.run(
        world_tensor_raw=world_tensor_raw,
        urgency=urgency,
        is_personal=is_personal,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=affinities,
        agent_memory=agent_memory,
    )

    # 4. Measure the "Information Block Rate"
    # eng is the raw engagement energy. If it's very close to 0, they blocked it.
    eng_np = eng.cpu().numpy()
    
    trad_eng = eng_np[:half]
    prog_eng = eng_np[half:]

    # Agents are considered "blocked" if their engagement is exactly 0
    trad_blocked = np.sum(trad_eng == 0.0) / half * 100
    prog_blocked = np.sum(prog_eng == 0.0) / half * 100

    print("\nResults:")
    print(f"Traditionalists (Low Openness) Block Rate: {trad_blocked:.2f}%")
    print(f"Progressives (High Openness) Block Rate: {prog_blocked:.2f}%")
    
    print(f"\nAverage Engagement (Traditionalists): {np.mean(trad_eng):.4f}")
    print(f"Average Engagement (Progressives): {np.mean(prog_eng):.4f}")

    # 5. Visualization
    plt.figure(figsize=(10, 6))

    plt.hist(prog_eng, bins=30, alpha=0.6, label='Progressives (High Openness)', color='blue')
    plt.hist(trad_eng, bins=30, alpha=0.6, label='Traditionalists (Low Openness)', color='red')

    plt.title('Cognitive Gate: Engagement with a Progressive Event', fontsize=14, fontweight='bold')
    plt.xlabel('Engagement Score (Energy)', fontsize=12)
    plt.ylabel('Number of Agents', fontsize=12)
    
    # Add text box with stats
    textstr = f"Traditionalists Blocked: {trad_blocked:.1f}%\nProgressives Blocked: {prog_blocked:.1f}%"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.gca().text(0.5, 0.5, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='center', horizontalalignment='center', bbox=props)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    output_path = os.path.join(os.path.dirname(__file__), "selective_exposure_gate.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved visualization to: {output_path}")

if __name__ == "__main__":
    test_cognitive_gate()
