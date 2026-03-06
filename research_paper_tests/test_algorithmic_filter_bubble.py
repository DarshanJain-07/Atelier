import numpy as np
import torch
import sys
import os

# Add parent directory to path to import schema, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import SimConfig, DIMENSIONS
from generate_society import generate_society
from cognitive_engine import CognitiveEngine

def test_algorithmic_filter_bubble():
    print("--- Testing Algorithmic 2-Pass Filter Bubble ---")
    
    # Initialize engine with algorithmic amplification ENABLED
    config_algo = SimConfig(
        num_agents=2000, 
        use_algorithmic_amplification=True,
        algo_sample_size=0.1, # 10% A/B Test
        algo_exaggeration_factor=2.0 # 2x Amplification for testing visibility
    )
    
    df_meta, exposures, personalities, affinities = generate_society(config_algo)
    
    # A seemingly "boring" event. 
    # Let's say it's a minor tech update that slightly touches Innovation and slightly touches Freedom.
    # Most people wouldn't care.
    world_tensor_boring = torch.zeros(1, 12)
    world_tensor_boring[0, 6] = 0.2  # Innovation
    world_tensor_boring[0, 7] = -0.1 # Freedom (slight threat)
    
    print("\n[ Scenario: Broadcasting a 'Boring' Tech Update ]")
    print(f"Original Tensor: Innovation: 0.2, Freedom: -0.1")
    
    # We simulate what would happen WITHOUT the algorithm first to get a baseline
    cog_engine = CognitiveEngine(config_algo)
    _, _, eng_baseline, _ = cog_engine.run(
        world_tensor_raw=world_tensor_boring,
        urgency=0.5,
        is_personal=False,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=affinities,
        agent_memory=None
    )
    
    # Now we simulate WITH the algorithm (Pass 1 -> Algorithm -> Pass 2)
    sample_size = int(config_algo.num_agents * config_algo.algo_sample_size)
    
    # PASS 1: The A/B Test
    _, ab_attention, ab_engagement, _ = cog_engine.run(
        world_tensor_raw=world_tensor_boring,
        urgency=0.5,
        is_personal=False,
        exposures=exposures[:sample_size],
        personalities=personalities[:sample_size],
        agent_affinities=affinities[:sample_size],
        agent_memory=None
    )
    
    # The Algorithm's Intervention
    engagement_weighted_attention = ab_attention * ab_engagement.unsqueeze(1)
    avg_attention_per_dim = engagement_weighted_attention.mean(dim=0)
    top_dims = torch.topk(avg_attention_per_dim, k=2).indices
    
    mutated_world_tensor = world_tensor_boring.clone()
    for dim_idx in top_dims:
        current_val = mutated_world_tensor[0, dim_idx].item()
        if abs(current_val) > 0.05:
            mutated_world_tensor[0, dim_idx] *= config_algo.algo_exaggeration_factor
        else:
            mutated_world_tensor[0, dim_idx] = -0.3 # Algorithm injects a threat
        
    mutated_world_tensor = torch.clamp(mutated_world_tensor, -1.0, 1.0)
    
    # PASS 2: Viral Broadcast
    _, _, eng_algo, _ = cog_engine.run(
        world_tensor_raw=mutated_world_tensor,
        urgency=0.5,
        is_personal=False,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=affinities,
        agent_memory=None
    )
    
    print("\n[ The Algorithm's Decision ]")
    print(f"The algorithm noticed high engagement in these dimensions:")
    for dim_idx in top_dims:
        print(f" - {DIMENSIONS[dim_idx]}")
        
    print("\n[ Final Mutated Event Broadcast to Society ]")
    print(f"Mutated Tensor: Innovation: {mutated_world_tensor[0, 6].item():.2f}, Freedom: {mutated_world_tensor[0, 7].item():.2f}")
    
    print("\n[ Results ]")
    print(f"Average Engagement (Without Algorithm): {eng_baseline.mean().item():.4f}")
    print(f"Average Engagement (With Algorithm):    {eng_algo.mean().item():.4f}")
    print(f"Total Engagement Increase:              {((eng_algo.mean().item() / eng_baseline.mean().item()) - 1) * 100:.1f}%")

if __name__ == "__main__":
    test_algorithmic_filter_bubble()
