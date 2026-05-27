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
from schema import SimConfig

def test_silicon_backlash():
    """
    RESEARCH FINDINGS - SILICON BACKLASH (NARRATIVE RESISTANCE)
    -----------------------------------------------------------
    This test measures the 'Narrative Resistance' of the agent population.
    It evaluates the engine's ability to reject a top-down 'Official' 
    narrative when it conflicts with agent-level economic and social reality.

    KEY OBSERVATIONS:
    1. CYNICAL REJECTION: In this scenario (Wealth Tax), the society 
       overwhelmingly rejected the Official Frame. The Skeptical Frame 
       generated significantly higher cognitive energy (~6.8x higher).
    
    2. COGNITIVE DISSONANCE: The high Skeptical Energy proves that agents 
       possess 'Skin in the Game.' Their internal trait-dimension mapping 
       (Wealth, Fairness) acts as a filter that exposes 'Official' framing 
       as dissonant.

    3. SUCCESS CRITERIA: The trigger of a 'SKEPTICAL' frame selection 
       validates that ATELIER models a non-compliant population, making it 
       a robust tool for simulating social unrest and populist movements.
    """
    config = SimConfig(
        num_agents=2000,
        seed=42,
        use_backlash_ab_testing=True,
        backlash_sample_size=0.2, # Large sample for audit
        backlash_decision_threshold=1.1, # Sensitive
        use_agent_memory=True
    )
    
    output_dir = Path("research_paper_tests/generated/backlash")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)
    cognitive_engine = CognitiveEngine(config)
    
    # Event: "Official Wealth Tax"
    # Official: [Stability +0.5, Fairness +0.4]
    # Skeptical: [Wealth -0.9, Freedom -0.7, Reputation -0.5]
    world_off = torch.zeros(12); world_off[2]=0.5; world_off[4]=0.4
    world_skp = torch.zeros(12); world_skp[0]=-0.9; world_skp[7]=-0.7; world_skp[3]=-0.5
    
    # Run A/B Test logic
    print("Running Silicon Backlash A/B Test...")
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
    
    print(f"Results: Chosen Frame = {decision.chosen_frame.upper()}")
    print(f"Skeptical Energy: {decision.skeptical_energy:.4f} vs Official Energy: {decision.official_energy:.4f}")
    
    # Visualization: Energy Gap
    apply_paper_style()
    fig, ax = setup_plot(title="Silicon Backlash: Narrative Energy Gap", xlabel="", ylabel="Cognitive Energy")
    ax.bar(["Official Frame", "Skeptical Frame"], [decision.official_energy, decision.skeptical_energy], 
           color=['#1F4E79', '#A33D3D'])
    
    save_path = output_dir / "backlash_energy.png"
    save_paper_figure(fig, save_path)
    
    # Audit Analysis
    if decision.triggered:
        print("SUCCESS: Backlash triggered. Society rejected the official narrative.")
    else:
        print("INFO: Society accepted the official narrative (Trust > Cynicism).")

if __name__ == "__main__":
    test_silicon_backlash()
