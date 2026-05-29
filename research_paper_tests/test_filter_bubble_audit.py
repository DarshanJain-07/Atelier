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
from research_paper_tests.config_schema import SimConfig
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater

def run_cascade_audit(use_algo: bool):
    """Runs a 10-cycle simulation with or without algorithmic amplification."""
    config = SimConfig(
        num_agents=2000,
        seed=42, # Will be overridden by run_monte_carlo's global seeds
        use_algorithmic_amplification=use_algo,
        algo_exaggeration_factor=2.0,
        homophily_strength=8.0, # Strong echo chambers
        use_agent_memory=True
    )
    
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)
    cognitive_engine = CognitiveEngine(config)
    physics_engine = SocialPhysicsEngine(config)
    
    N = config.num_agents
    influence = torch.tensor(df_meta["Influence"].values, dtype=torch.float32)
    agent_memory = torch.zeros(N, 12)
    
    # Sequence of 10 events: Each slightly ambiguous
    base_event = torch.zeros(12); base_event[0]=0.1; base_event[2]=-0.1; base_event[4]=-0.1
    
    pol_history = []
    
    for i in range(10):
        current_signal = base_event.clone()
        if use_algo and i > 0:
            current_signal[4] *= (1.0 + 0.5 * i) # Scale Fairness (Negative)
            current_signal[2] *= (1.0 + 0.3 * i) # Scale Stability (Negative)

        context_vector, _, engagement_scores = cognitive_engine.run(
            world_tensor_raw=current_signal.unsqueeze(0),
            urgency=0.5,
            is_personal=False,
            exposures=exposures,
            personalities=personalities,
            agent_affinities=affinities,
            agent_memory=agent_memory,
            adjacency_matrix=adjacency_matrix
        )
        
        emotion_tensor = cognitive_engine.project_emotions(context_vector)
        result = physics_engine.aggregate_society(
            emotion_tensor=emotion_tensor,
            influence_scores=influence,
            engagement_scores=engagement_scores,
            adjacency_matrix=adjacency_matrix,
            personalities=personalities
        )
        
        agent_memory = cognitive_engine.consolidate_memory(
            agent_memory, context_vector, social_rehearsal_factor=result["acting_ratio"]
        )
        pol_history.append(result["polarization"])
        
    return pol_history

def test_filter_bubble_audit(n_seeds):
    """
    RESEARCH FINDINGS - ALGORITHMIC FILTER BUBBLE AUDIT
    ---------------------------------------------------
    Uses statistical validation to confirm that algorithmic amplification 
    accelerates polarization significantly beyond linear homophily.
    """
    output_dir = Path("research_paper_tests/generated/filter_bubble")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def runner():
        control_pol = run_cascade_audit(use_algo=False)
        treatment_pol = run_cascade_audit(use_algo=True)
        
        growth_control = control_pol[-1] - control_pol[0]
        growth_treatment = treatment_pol[-1] - treatment_pol[0]
        
        return {
            "growth_control": growth_control,
            "growth_treatment": growth_treatment,
            "control_pol": control_pol,
            "treatment_pol": treatment_pol
        }

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    
    control_growths = [r["growth_control"] for r in results]
    treatment_growths = [r["growth_treatment"] for r in results]
    
    # Statistical Validation
    assert_statistically_greater(treatment_growths, control_growths)
    
    # Visualization (using the first run's data)
    first_run = results[0]
    apply_paper_style()
    fig, ax = setup_plot(
        title="Filter Bubble Audit: Algorithmic Amplification Loop",
        xlabel="Simulation Cycles",
        ylabel="Polarization (Bimodality)"
    )
    
    cycles = np.arange(1, 11)
    ax.plot(cycles, first_run["control_pol"], marker='o', label='Linear Echo Chambers (No Algo)', color='#7A7A7A')
    ax.plot(cycles, first_run["treatment_pol"], marker='s', label='Algorithmic Feed (With Algo)', color='#C55A11')
    
    ax.legend()
    save_path = output_dir / "filter_bubble_hockey_stick.png"
    save_paper_figure(fig, save_path)
    
    print(f"\nSaved plot to {save_path}")
    
    mean_control = np.mean(control_growths)
    mean_treatment = np.mean(treatment_growths)
    acceleration = mean_treatment / (mean_control + 1e-9)
    print(f"Mean Acceleration Factor: {acceleration:.2f}x")

if __name__ == "__main__":
    # For manual execution, use a default n_seeds
    test_filter_bubble_audit(n_seeds=5)
