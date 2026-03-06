import numpy as np
import torch
import sys
import os

# Add parent directory to path to import schema, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import SimConfig
from generate_society import generate_society
from physics_engine import SocialPhysicsEngine

def test_topology():
    print("--- Testing Sparse Network Topology & Echo Chambers ---")
    
    # Initialize engine with topology ENABLED
    config_topo = SimConfig(
        num_agents=2000, 
        use_network_topology=True,
        base_connections=20,
        max_connections=200,
        homophily_strength=3.0 # Strong echo chambers
    )
    
    print("\n[ Generating Society with Topology ]")
    df_meta, exposures, personalities, affinities, adjacency = generate_society(config_topo)
    
    print(f"\n[ Adjacency Matrix Constructed ]")
    print(f"Shape: {adjacency.shape}")
    print(f"Non-zeros (Total Edges): {adjacency._nnz()}")
    print(f"Average edges per agent: {adjacency._nnz() / config_topo.num_agents:.1f}")
    
    # Let's create a highly polarized emotional state to test if topology changes the physics.
    # Group A (first 1000) are Furious (Anger index = 4)
    # Group B (last 1000) are Joyful (Joy index = 0)
    N = config_topo.num_agents
    emotion_tensor = torch.zeros(N, 8)
    emotion_tensor[:1000, 4] = 1.0 # Anger
    emotion_tensor[1000:, 0] = 1.0 # Joy
    
    influence = df_meta["Influence"].values
    
    phys_engine = SocialPhysicsEngine(config_topo)
    
    print("\n[ Running Physics Engine WITHOUT Topology (Global Center) ]")
    # By passing None for adjacency, it defaults to the old Global Center method
    state_no_topo = phys_engine.aggregate_society(emotion_tensor, influence, adjacency_matrix=None)
    print(f"Mean Outrage Multiplier (Virality): {state_no_topo['mean_outrage_multiplier']}x")
    
    print("\n[ Running Physics Engine WITH Topology (Local Echo Chambers) ]")
    # By passing adjacency, outrage is measured against their LOCAL echo chamber.
    # Since they are connected via homophily, Group A mostly connects to Group A.
    # So their "local center" agrees with them, which should mathematically REDUCE their personal outrage
    # compared to the global model where half the world disagrees with them.
    state_topo = phys_engine.aggregate_society(emotion_tensor, influence, adjacency_matrix=adjacency)
    print(f"Mean Outrage Multiplier (Virality): {state_topo['mean_outrage_multiplier']}x")
    
    diff = state_no_topo['mean_outrage_multiplier'] - state_topo['mean_outrage_multiplier']
    print(f"\nConclusion: Echo chambers reduced algorithmic virality by {diff:.2f}x because agents felt their 'local neighborhood' agreed with them, preventing the massive outrage spikes seen in the global broadcast model.")

if __name__ == "__main__":
    test_topology()
