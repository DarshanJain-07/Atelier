import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from schema import SimConfig, DIMENSIONS, PSYCH_PROJECTION, EMOTION_LABELS
from cognitive_engine import CognitiveEngine
from physics_engine import SocialPhysicsEngine

def run_simulation(wealth_event, fairness_event, config):
    idx_wealth = DIMENSIONS.index("Wealth")
    idx_fairness = DIMENSIONS.index("Fairness")
    
    # We create two distinct groups of 100 agents each.
    exposures_marginalized = torch.zeros(100, 12)
    exposures_marginalized[:, idx_wealth] = -0.8
    exposures_marginalized[:, idx_fairness] = -0.8
    
    personalities_marginalized = torch.ones(100, 5) * 0.5
    personalities_marginalized[:, 3] = 0.1 # Low Agreeableness
    personalities_marginalized[:, 4] = 0.9 # High Neuroticism
    
    exposures_elites = torch.zeros(100, 12)
    exposures_elites[:, idx_wealth] = 0.8
    exposures_elites[:, idx_fairness] = 0.8
    
    personalities_elites = torch.ones(100, 5) * 0.5
    personalities_elites[:, 3] = 0.9 # High Agreeableness
    personalities_elites[:, 4] = 0.1 # Low Neuroticism
    
    exposures = torch.cat([exposures_marginalized, exposures_elites], dim=0)
    personalities = torch.cat([personalities_marginalized, personalities_elites], dim=0)
    agent_affinities = torch.ones(200, 12)
    
    world_tensor_raw = torch.zeros(1, 12)
    world_tensor_raw[0, idx_wealth] = wealth_event
    world_tensor_raw[0, idx_fairness] = fairness_event
    
    cog_engine = CognitiveEngine(config)
    physics_engine = SocialPhysicsEngine(config)
    
    context_vector, attention_weights, engagement_scores, _ = cog_engine.run(
        world_tensor_raw=world_tensor_raw,
        urgency=0.0,
        is_personal=False,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=agent_affinities,
        agent_memory=None
    )
    
    emotion_tensor = torch.matmul(context_vector, PSYCH_PROJECTION.to(context_vector.device))
    
    # Group results
    marg_emotions = emotion_tensor[:100]
    elite_emotions = emotion_tensor[100:]
    marg_dom = EMOTION_LABELS[torch.mode(torch.argmax(marg_emotions, dim=1)).values.item()]
    elite_dom = EMOTION_LABELS[torch.mode(torch.argmax(elite_emotions, dim=1)).values.item()]
    
    # Use Physics Engine to get society aggregate
    influence_scores = torch.ones(200) # equal influence
    agg_result = physics_engine.aggregate_society(
        emotion_tensor, 
        influence_scores,
        engagement_scores
    )
    
    return marg_dom, elite_dom, agg_result["dominant_emotion"]

def test_relative_deprivation_grid_search():
    print("--- Running Test: Relative Deprivation (RDE) & Physics Engine Grid Search ---")
    
    config = SimConfig(
        num_agents=200,
        use_signal_distortion=False,
        use_time_pressure=False,
    )
    
    print(f"{'Wealth Gain':<15} | {'Fairness Drop':<15} | {'Marginalized':<15} | {'Elites':<15} | {'Society Agg':<15}")
    print("-" * 80)
    
    # We will test a constant objective Wealth gain of +0.5, but varying the perceived fairness
    wealth_gain = 0.5
    for fairness_drop in np.linspace(0.0, -1.0, 11): # from 0.0 down to -1.0
        marg_dom, elite_dom, agg_dom = run_simulation(wealth_gain, fairness_drop, config)
        print(f"{wealth_gain:<15.2f} | {fairness_drop:<15.2f} | {marg_dom:<15} | {elite_dom:<15} | {agg_dom:<15}")
        
    print("\n✅ Grid Search completed successfully, physics engine integrated.")

if __name__ == "__main__":
    test_relative_deprivation_grid_search()
