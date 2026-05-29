import json
from pathlib import Path
from itertools import product
import torch
import numpy as np
import pandas as pd

from cognitive_engine import CognitiveEngine
from generate_society import generate_society
from physics_engine import SocialPhysicsEngine
from research_paper_tests.config_schema import SimConfig
from research_paper_tests.stats_utils import run_monte_carlo, assert_monotonic_relationship

def run_single_hysteresis_simulation(config: SimConfig):
    """Runs a fixed 15-step hysteresis sim and returns the ratio."""
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)
    cognitive_engine = CognitiveEngine(config)
    physics_engine = SocialPhysicsEngine(config)
    
    N = config.num_agents
    agent_memory = torch.zeros(N, 12)
    influence = torch.tensor(df_meta["Influence"].values, dtype=torch.float32)
    
    polarization_history = []
    
    # Define Events (Moderate Unifying Signal after Outrage)
    outrage = torch.zeros(12); outrage[1]=-0.8; outrage[2]=-0.7; outrage[4]=-0.9
    unifying = torch.zeros(12); unifying[0]=0.6; unifying[1]=0.6; unifying[4]=0.6; unifying[9]=0.6
    events = [outrage] * 5 + [unifying] * 10
    
    for i, event_signal in enumerate(events):
        context_vector, _, engagement_scores = cognitive_engine.run(
            world_tensor_raw=event_signal.unsqueeze(0),
            urgency=0.5, is_personal=False,
            exposures=exposures, personalities=personalities,
            agent_affinities=affinities, agent_memory=agent_memory,
            adjacency_matrix=adjacency_matrix
        )
        emotion_tensor = cognitive_engine.project_emotions(context_vector)
        result = physics_engine.aggregate_society(
            emotion_tensor=emotion_tensor, influence_scores=influence,
            engagement_scores=engagement_scores, adjacency_matrix=adjacency_matrix,
            personalities=personalities
        )
        polarization_history.append(result["polarization"])
        agent_memory = cognitive_engine.consolidate_memory(
            agent_memory, context_vector, social_rehearsal_factor=result["acting_ratio"]
        )

    initial = polarization_history[0]
    peak = max(polarization_history)
    final = polarization_history[-1]
    
    total_push = peak - initial
    recovery = peak - final
    
    if total_push < 0.05:
        return 0.0
        
    ratio = 1.0 - (recovery / total_push)
    return float(max(0.0, min(1.0, ratio)))

def test_hysteresis_grid_search(n_seeds):
    """
    RESEARCH FINDINGS - HYSTERESIS & SOCIAL INERTIA
    ----------------------------------------------
    Uses statistical validation to find the tipping point where 
    societal memory prevents recovery from polarization.
    """
    
    # Define the sweep across decay rates
    decay_rates = [0.70, 0.75, 0.80, 0.85, 0.90]
    results_summary = []
    
    print(f"Starting Statistical Hysteresis Sweep: {len(decay_rates)} levels...")
    
    mean_ratios = []
    
    for decay_rate in decay_rates:
        def runner():
            config = SimConfig(
                num_agents=500, # Faster for grid search
                use_agent_memory=True,
                memory_decay_rate=decay_rate,
                memory_reconciliation_gain=0.5,
                stewing_ticks=5
            )
            return run_single_hysteresis_simulation(config)
            
        ratios = run_monte_carlo(runner, n_seeds=n_seeds)
        mean_ratio = np.mean(ratios)
        mean_ratios.append(mean_ratio)
        
        results_summary.append({
            "memory_decay_rate": decay_rate,
            "mean_hysteresis_ratio": round(float(mean_ratio), 4),
            "std_error": round(float(np.std(ratios) / np.sqrt(n_seeds)), 4)
        })
        print(f"Decay Rate {decay_rate}: Mean Ratio = {mean_ratio:.4f}")

    # Save to CSV for analysis
    output_dir = Path("research_paper_tests/generated/hysteresis_grid")
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results_summary)
    df.to_csv(output_dir / "grid_statistical_results.csv", index=False)
    
    # Statistical Validation: 
    # Increasing decay rate (faster forgetting) should decrease hysteresis (easier recovery)
    # This means a negative correlation
    assert_monotonic_relationship(
        decay_rates, 
        mean_ratios, 
        expected_direction="negative"
    )

if __name__ == "__main__":
    test_hysteresis_grid_search(n_seeds=3)
