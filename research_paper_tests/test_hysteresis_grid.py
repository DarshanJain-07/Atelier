import json
from pathlib import Path
from itertools import product
import torch
import numpy as np
import pandas as pd

from cognitive_engine import CognitiveEngine
from generate_society import generate_society
from physics_engine import SocialPhysicsEngine
from schema import SimConfig

def run_single_hysteresis_simulation(config: SimConfig):
    """Runs a fixed 10-step hysteresis sim and returns the ratio."""
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)
    cognitive_engine = CognitiveEngine(config)
    physics_engine = SocialPhysicsEngine(config)
    
    N = config.num_agents
    agent_memory = torch.zeros(N, 12)
    influence = torch.tensor(df_meta["Influence"].values, dtype=torch.float32)
    
    polarization_history = []
    
    # Define Events (Moderate Unifying Signal)
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
        
        if config.memory_decay_rate == 0.4 and config.memory_reconciliation_gain == 0.8:
            print(f"  Step {i+1}: Pol={result['polarization']:.3f}, Val={result['sentiment_valence']:.3f}, Viral_Mul={result['mean_outrage_multiplier']:.3f}, Act={result['acting_ratio']:.3f}")

    initial = polarization_history[0]
    peak = max(polarization_history)
    final = polarization_history[-1]
    
    total_push = peak - initial
    recovery = peak - final
    
    # If it didn't push much, ratio is 0
    if total_push < 0.05:
        return 0.0, float(final)
        
    ratio = 1.0 - (recovery / total_push)
    # Clamp ratio between 0 and 1
    ratio = max(0.0, min(1.0, ratio))
    return float(ratio), float(final)

def test_hysteresis_grid_search():
    """Systematically sweeps parameters to find the optimal social inertia."""
    
    # Define the Grid (High precision around the tipping point)
    grid = {
        "memory_decay_rate": [0.77, 0.78, 0.79, 0.80, 0.81],
        "memory_reconciliation_gain": [0.5],
        "stewing_ticks": [5]
    }
    
    # Generate all combinations
    keys = list(grid.keys())
    combinations = list(product(*grid.values()))
    
    results = []
    output_dir = Path("research_paper_tests/generated/hysteresis_grid")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Grid Search: {len(combinations)} combinations...")
    
    for i, vals in enumerate(combinations):
        params = dict(zip(keys, vals))
        
        # Setup config with grid params
        config = SimConfig(
            num_agents=1000, # Smaller for speed
            seed=42,
            use_agent_memory=True,
            **params
        )
        
        ratio, final_pol = run_single_hysteresis_simulation(config)
        
        res = params.copy()
        res["hysteresis_ratio"] = round(ratio, 4)
        res["final_polarization"] = round(final_pol, 4)
        results.append(res)
        
        print(f"Trial {i+1}/{len(combinations)}: {params} -> Ratio: {ratio:.4f}")

    # Save to CSV for analysis
    df = pd.DataFrame(results)
    csv_path = output_dir / "grid_results.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\nGrid Search Complete. Results saved to {csv_path}")
    
    # Automated Anomaly Detection
    lock_in_cases = df[df["hysteresis_ratio"] > 0.9]
    elastic_cases = df[df["hysteresis_ratio"] < 0.2]
    
    if not lock_in_cases.empty:
        print(f"ALERT: Found {len(lock_in_cases)} combinations resulting in permanent lock-in.")
    if not elastic_cases.empty:
        print(f"ALERT: Found {len(elastic_cases)} combinations resulting in excessive elasticity.")

if __name__ == "__main__":
    test_hysteresis_grid_search()
