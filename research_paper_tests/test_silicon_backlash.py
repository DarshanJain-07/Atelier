import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from cognitive_engine import CognitiveEngine
from generate_society import generate_society
from physics_engine import SocialPhysicsEngine
from research_paper_tests.plotting_utils import apply_paper_style, save_paper_figure, setup_plot
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater
from research_paper_tests.config_schema import SimConfig

def test_silicon_backlash():
    """
    RESEARCH FINDINGS - SILICON BACKLASH (NARRATIVE RESISTANCE)
    -----------------------------------------------------------
    This test measures the 'Narrative Resistance' of the agent population.
    It evaluates the engine's ability to reject a top-down 'Official' 
    narrative when it conflicts with agent-level economic and social reality.

    STATISTICAL VALIDATION:
    Uses Monte Carlo seeds (default 5) and Welch's t-test to ensure the 
    Skeptical energy is significantly higher than the Official energy.
    """
    config = SimConfig(
        num_agents=1000, # Sufficient for statistical significance
        use_backlash_ab_testing=True,
        backlash_sample_size=0.2, 
        backlash_decision_threshold=1.1, 
        use_agent_memory=True
    )
    
    output_dir = Path("research_paper_tests/generated/backlash")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_backlash_trial():
        _, exposures, personalities, affinities, adjacency_matrix = generate_society(config)
        cognitive_engine = CognitiveEngine(config)
        
        # Event: "Official Wealth Tax"
        # Official: [Stability +0.5, Fairness +0.4]
        # Skeptical: [Wealth -0.9, Freedom -0.7, Reputation -0.5]
        world_off = torch.zeros(12); world_off[2]=0.5; world_off[4]=0.4
        world_skp = torch.zeros(12); world_skp[0]=-0.9; world_skp[7]=-0.7; world_skp[3]=-0.5
        
        decision = cognitive_engine.run_backlash_ab_test(
            world_tensor_off=world_off.unsqueeze(0),
            world_tensor_skp=world_skp.unsqueeze(0),
            backlash_potential=1.5, # High tension
            urgency=0.6,
            is_personal=False,
            exposures=exposures,
            personalities=personalities,
            agent_affinities=affinities,
            adjacency_matrix=adjacency_matrix
        )
        return decision.official_energy, decision.skeptical_energy

    print("Running Silicon Backlash Monte Carlo trials...")
    results = run_monte_carlo(run_backlash_trial)
    official_energies = [r[0] for r in results]
    skeptical_energies = [r[1] for r in results]
    
    # Statistical validation
    assert_statistically_greater(skeptical_energies, official_energies)
    
    # Visualization: Energy Gap (using means and standard errors)
    apply_paper_style()
    avg_official = np.mean(official_energies)
    avg_skeptical = np.mean(skeptical_energies)
    std_official = np.std(official_energies)
    std_skeptical = np.std(skeptical_energies)
    
    fig, ax = setup_plot(title="Silicon Backlash: Narrative Energy Gap (MC)", xlabel="", ylabel="Cognitive Energy")
    ax.bar(["Official Frame", "Skeptical Frame"], [avg_official, avg_skeptical], 
           yerr=[std_official, std_skeptical], capsize=5,
           color=['#1F4E79', '#A33D3D'])
    
    save_path = output_dir / "backlash_energy_mc.png"
    save_paper_figure(fig, save_path)
    plt.close(fig)
    
    print(f"Results: Skeptical Mean = {avg_skeptical:.4f} vs Official Mean = {avg_official:.4f} (p < 0.05)")

if __name__ == "__main__":
    test_silicon_backlash()
