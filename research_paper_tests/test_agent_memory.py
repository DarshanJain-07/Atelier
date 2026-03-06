import numpy as np
import torch
import sys
import os

# Add parent directory to path to import schema, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import SimConfig
from generate_society import generate_society
from cognitive_engine import CognitiveEngine

def test_agent_memory():
    print("--- Testing Agent Memory (Extreme Desensitization & Trigger Stacking Loop) ---")
    
    # Initialize engine with memory enabled
    config = SimConfig(
        num_agents=1000, 
        use_agent_memory=True,
        memory_desensitization_gain=5.0, # MASSIVE fatigue multiplier
        memory_trigger_stacking_gain=15.0  # MASSIVE stacking multiplier
    )
    
    df_meta, exposures, personalities, affinities = generate_society(config)
    cog_engine = CognitiveEngine(config)
    
    # Event A: A massive wealth threat (The repeated shock)
    world_tensor_threat = torch.zeros(1, 12)
    world_tensor_threat[0, config.wealth_dim_idx] = -0.8 # -0.8 Wealth (Threat)
    
    print("\n[ Running 50 Iterations of the Same Wealth Threat (-0.8) ]")
    # Start with empty memory
    agent_memory = torch.zeros_like(exposures)
    
    for i in range(50):
        ctx, att, eng, agent_memory = cog_engine.run(
            world_tensor_raw=world_tensor_threat,
            urgency=0.5,
            is_personal=False,
            exposures=exposures,
            personalities=personalities,
            agent_affinities=affinities,
            agent_memory=agent_memory
        )
        if i == 0:
            print(f"Iteration 1 (Fresh): Average Engagement = {eng.mean().item():.4f}")
        elif i == 49:
            print(f"Iteration 50 (Fatigued): Average Engagement = {eng.mean().item():.4f}")

    print("\n[ Society is now heavily desensitized to Wealth Threats but carries massive internal stress ]")
    
    print("\n[ The Spark: A Minor New Threat (Physical Safety -0.2) ]")
    # Because they are already stressed from 50 iterations of wealth loss, a minor new threat might explode
    world_tensor_new_threat = torch.zeros(1, 12)
    world_tensor_new_threat[0, 1] = -0.2 # -0.2 Safety (Minor Threat)
    
    ctx_stacked, att_stacked, eng_stacked, mem_stacked = cog_engine.run(
        world_tensor_raw=world_tensor_new_threat,
        urgency=0.5,
        is_personal=False,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=affinities,
        agent_memory=agent_memory  # Pass the accumulated stress from 50 iterations
    )
    print(f"Average Engagement (Trigger Stacked): {eng_stacked.mean().item():.4f}")
    
    # Let's run the exact same Minor Threat on a FRESH society without memory to prove it was stacked
    print("\n[ Baseline: Same Minor Safety Threat on FRESH Society (No Memory) ]")
    ctx_base, att_base, eng_base, _ = cog_engine.run(
        world_tensor_raw=world_tensor_new_threat,
        urgency=0.5,
        is_personal=False,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=affinities,
        agent_memory=torch.zeros_like(exposures)
    )
    print(f"Average Engagement (Baseline / Calm): {eng_base.mean().item():.4f}")

if __name__ == "__main__":
    test_agent_memory()
