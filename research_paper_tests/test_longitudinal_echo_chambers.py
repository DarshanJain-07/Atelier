import torch
import numpy as np
import pytest
from generate_society import generate_society
from society_evolution import SocietyEvolution
from schema import SimConfig
from research_paper_tests._metrics import gini
from networkx.algorithms.community import louvain_communities
import networkx as nx

def test_wealth_gini_correlates_with_modularity():
    """
    Test 2: Longitudinal Echo Chambers (Generational Drift)
    Goal: Validate that economic inequality (Wealth Gini) drives topological balkanization (Modularity).
    """
    # 1. Setup two scenarios: Low Inequality and High Inequality
    base_config = SimConfig()
    base_config.num_agents = 500
    base_config.evolution_generations = 5
    base_config.use_network_topology = True
    
    # --- Scenario A: Low Gini (Redistributive) ---
    low_gini_config = SimConfig()
    # Copy base settings
    for field in low_gini_config.__dataclass_fields__:
        setattr(low_gini_config, field, getattr(base_config, field))
    low_gini_config.inheritance_fraction = 0.1 # High redistribution
    low_gini_config.base_return_rate = 0.01
    
    # --- Scenario B: High Gini (Dynastic) ---
    high_gini_config = SimConfig()
    for field in high_gini_config.__dataclass_fields__:
        setattr(high_gini_config, field, getattr(base_config, field))
    high_gini_config.inheritance_fraction = 0.9 # High inheritance
    high_gini_config.base_return_rate = 0.1 # High growth for the wealthy
    
    def run_scenario(config):
        # Generate initial society
        metadata, exposures, personalities, affinities, adjacency = generate_society(config)
        
        # Initial Gini
        initial_wealth = torch.tensor(metadata["Raw_Wealth"].values)
        initial_gini = gini(initial_wealth)
        
        # Evolve
        evolution = SocietyEvolution(config, metadata, exposures, personalities)
        evolved_metadata, evolved_exposures, evolved_personalities = evolution.evolve()
        
        # Final Gini
        final_wealth = torch.tensor(evolved_metadata["Raw_Wealth"].values)
        final_gini = gini(final_wealth)
        
        # Final Modularity (using NetworkX for Louvain)
        # Note: In ATELIER, adjacency is generated based on exposures. 
        # For the test, we regenerate the topology based on the EVOLVED exposures.
        from generate_society import create_topology
        evolved_adjacency = create_topology(
            config, 
            evolved_exposures, 
            evolved_personalities,
            influence_scores=evolved_metadata["Influence"].values,
            raw_wealth=evolved_metadata["Raw_Wealth"].values
        )
        
        # Convert sparse to NX
        indices = evolved_adjacency.coalesce().indices()
        G = nx.Graph()
        G.add_nodes_from(range(config.num_agents))
        G.add_edges_from(indices.T.tolist())
        
        communities = louvain_communities(G, seed=config.seed)
        modularity = nx.community.modularity(G, communities)
        
        return final_gini, modularity

    low_gini, low_mod = run_scenario(low_gini_config)
    high_gini, high_mod = run_scenario(high_gini_config)
    
    print(f"Low Gini Scenario: Gini={low_gini:.3f}, Modularity={low_mod:.3f}")
    print(f"High Gini Scenario: Gini={high_gini:.3f}, Modularity={high_mod:.3f}")
    
    # Validation: High Gini should generally lead to higher modularity
    # (assuming class-based connection logic is working in generate_society.py)
    assert high_gini > low_gini
    # Relaxed assertion: If high gini doesn't lead to higher modularity, it flags a 
    # potential "weak connection" bug as per the user's request.
    if high_mod <= low_mod:
        pytest.fail(f"Balkanization failure: High Gini ({high_gini:.3f}) did not increase modularity ({high_mod:.3f}) vs Low Gini ({low_mod:.3f})")

if __name__ == "__main__":
    test_wealth_gini_correlates_with_modularity()
