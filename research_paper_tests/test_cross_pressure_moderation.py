import torch
import numpy as np
import networkx as nx
from generate_society import generate_society
from research_paper_tests.config_schema import PSYCH_PROJECTION, SimConfig
from cognitive_engine import CognitiveEngine
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater

def test_diplomat_effect_cross_pressure_moderation(n_seeds):
    """
    Test 4: Cross-Pressure Moderation (The "Diplomat" Effect)
    Goal: Validate that bridging-tie agents (high betweenness) are more "rational" (higher logic consistency).
    """
    config = SimConfig()
    config.num_agents = 300
    config.use_network_topology = True
    config.homophily_strength = 5.0 # Create clear clusters
    
    def runner():
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
        
        # Simulate distortion/perception
        distorted_signals = cognitive.distort_signal(polarizing_signal, personalities)
        
        # Calculate "Logic Consistency" or "Arousal Stability"
        # We'll assert diplomats have LOWER average arousal to a polarizing signal
        # because they integrate conflicting signals better (the 'diplomat' effect).
        
        emotions = torch.matmul(distorted_signals, PSYCH_PROJECTION)
        arousal = torch.norm(emotions, dim=1)
        
        diplomat_arousal = arousal[diplomat_indices].mean().item()
        isolated_arousal = arousal[isolated_indices].mean().item()
        
        return diplomat_arousal, isolated_arousal

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    diplomat_dist = [r[0] for r in results]
    isolated_dist = [r[1] for r in results]
    
    # Validation: Isolated agents should have higher arousal (less moderation) than diplomats
    assert_statistically_greater(isolated_dist, diplomat_dist)

if __name__ == "__main__":
    # For manual testing
    from research_paper_tests.stats_utils import get_monte_carlo_seeds
    test_diplomat_effect_cross_pressure_moderation(get_monte_carlo_seeds())
