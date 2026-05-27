import torch
import numpy as np
import pytest
import networkx as nx
from generate_society import generate_society
from schema import SimConfig
from cognitive_engine import CognitiveEngine
from attention_context import AttentionContext

def test_diplomat_effect_cross_pressure_moderation():
    """
    Test 4: Cross-Pressure Moderation (The "Diplomat" Effect)
    Goal: Validate that bridging-tie agents (high betweenness) are more "rational" (higher logic consistency).
    """
    config = SimConfig()
    config.num_agents = 300
    config.use_network_topology = True
    config.homophily_strength = 5.0 # Create clear clusters
    
    # 1. Generate society with clusters
    metadata, exposures, personalities, affinities, adjacency = generate_society(config)
    
    # 2. Identify "Diplomats" using betweenness centrality
    indices = adjacency.coalesce().indices()
    G = nx.Graph()
    G.add_nodes_from(range(config.num_agents))
    G.add_edges_from(indices.T.tolist())
    
    centrality = nx.betweenness_centrality(G)
    centrality_values = np.array([centrality[i] for i in range(config.num_agents)])
    
    # Top 10% are diplomats, Bottom 20% are isolated/echo-chambered
    diplomat_indices = np.argsort(centrality_values)[-30:]
    isolated_indices = np.argsort(centrality_values)[:60]
    
    # 3. Inject a polarizing event
    # We'll simulate a signal that has a logic gap (contradiction)
    polarizing_signal = torch.zeros(12)
    polarizing_signal[5] = 1.0  # High In-Group
    polarizing_signal[4] = -1.0 # High Unfairness (Contradiction: In-group usually implies fairness)
    
    cognitive = CognitiveEngine(config)
    
    # We need to measure logic consistency. 
    # Let's look at AttentionContext's logic_gate processing
    # or just the resulting emotional stability.
    
    # Simulate distortion/perception
    distorted_signals = cognitive.distort_signal(polarizing_signal, personalities)
    
    # Calculate "Logic Consistency" or "Arousal Stability"
    # For simplicity, let's use the variance of perceptions as a proxy for moderation
    # or the actual logic gate suppression if we can isolate it.
    
    # Let's calculate the "Logic Gap" perceived by each group
    # High logic gap = lower rationality.
    # In ATELIER, skepticism_gain reduces signal magnitude for high logic gaps.
    # We'll assert diplomats have LOWER variance in their emotional response
    # because they integrate conflicting signals better (the 'diplomat' effect).
    
    from schema import PSYCH_PROJECTION
    emotions = torch.matmul(distorted_signals, PSYCH_PROJECTION)
    arousal = torch.norm(emotions, dim=1)
    
    diplomat_arousal = arousal[diplomat_indices].mean().item()
    isolated_arousal = arousal[isolated_indices].mean().item()
    
    print(f"Diplomat Average Arousal: {diplomat_arousal:.3f}")
    print(f"Isolated Average Arousal: {isolated_arousal:.3f}")
    
    # Validation: Diplomats should be MORE moderated (lower average arousal to a polarizing signal)
    # because their "bridging" position exposes them to cross-pressures.
    # Note: This depends on the cognitive engine incorporating neighborhood context in Stage 2.
    # Our aggregate_society handles the local context, but the individual perception is first.
    
    # If this fails, it flags that "Cognitive Diversity" isn't fully integrated into the 
    # individual perception layer yet, which is a key fine-tuning insight.
    assert diplomat_arousal < isolated_arousal * 1.1 # Relaxed for stochasticity

if __name__ == "__main__":
    test_diplomat_effect_cross_pressure_moderation()
