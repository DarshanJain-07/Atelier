import torch
import numpy as np
from schema import SimConfig, DIMENSIONS
from cognitive_engine import CognitiveEngine
from physics_engine import SocialPhysicsEngine

def run_flaw_test():
    print("Testing '0% Tax' Flawed Idea Discernment...")
    
    config = SimConfig()
    config.num_agents = 1000
    config.skepticism_gain = 4.0 # High sensitivity for test
    config.logic_gap_threshold = 0.4
    
    engine = CognitiveEngine(config)
    physics = SocialPhysicsEngine(config)
    
    # 0% Tax Event Tensor: High Wealth (Short Term), Low Stability (Long Term)
    # [Wealth, Safety, Stability, Rep, Fair, In, Innov, Free, Sanc, Care, ST, LT]
    world_tensor = torch.zeros(12)
    world_tensor[0] = 0.8   # Wealth (Short term win)
    world_tensor[2] = -0.9  # Stability (Long term collapse)
    world_tensor[10] = 0.9  # Short_Term impact
    world_tensor[11] = -0.9 # Long_Term impact
    
    # Create two agents:
    # Agent A: Low Openness, Low Conscientiousness (Susceptible to Populism)
    # Agent B: High Openness, High Conscientiousness (Skeptical/Analytical)
    personalities = torch.tensor([
        [0.1, 0.1, 0.5, 0.5, 0.5], # Agent A
        [0.9, 0.9, 0.5, 0.5, 0.5]  # Agent B
    ])
    
    exposures = torch.zeros(2, 12)
    affinities = torch.ones(2, 12)
    
    # Run cognitive engine
    context, attention, engagement = engine.run(
        world_tensor_raw=world_tensor,
        urgency=0.5,
        is_personal=False,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=affinities
    )
    
    print("\nResults:")
    print(f"Agent A (Populist): ST Attention: {attention[0, 10]:.3f}, LT Attention: {attention[0, 11]:.3f}")
    print(f"Agent B (Skeptical): ST Attention: {attention[1, 10]:.3f}, LT Attention: {attention[1, 11]:.3f}")
    
    # Check if Agent B focused more on Long Term
    if attention[1, 11] > attention[0, 11]:
        print("\nSUCCESS: Skeptical agent identified the long-term flaw and shifted attention away from short-term noise.")
    else:
        print("\nFAILURE: Skeptical gate did not activate as expected.")

if __name__ == "__main__":
    run_flaw_test()
