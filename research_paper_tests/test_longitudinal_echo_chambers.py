import torch
import numpy as np
import pytest
import networkx as nx
from generate_society import generate_society, create_topology
from society_evolution import SocietyEvolution
from research_paper_tests.config_schema import SimConfig
from research_paper_tests._metrics import gini
from research_paper_tests.stats_utils import (
    run_monte_carlo,
    assert_monotonic_relationship,
    assert_statistically_greater
)

def test_wealth_gini_correlates_with_modularity(n_seeds):
    """
    Validation: Economic inequality (Wealth Gini) must drive topological balkanization (Modularity).
    We sweep the inheritance fraction and assert a monotonic increase in modularity 
    across multiple seeds, proving the "Economic Balkanization" law.
    """
    inheritance_sweep = [0.1, 0.4, 0.7, 0.9]
    mean_modularities = []
    mean_ginis = []

    def get_sim_runner(inheritance):
        def runner():
            config = SimConfig()
            config.num_agents = 200 # Scaled down for test performance
            config.evolution_generations = 3
            config.use_network_topology = True
            config.inheritance_fraction = inheritance
            config.base_return_rate = 0.05
            
            # 1. Initial State
            metadata, exposures, personalities, affinities, adjacency = generate_society(config)
            
            # 2. Evolve
            evolution = SocietyEvolution(config, metadata, exposures, personalities)
            evolved_metadata, evolved_exposures, evolved_personalities = evolution.evolve()
            
            # 3. Final Gini
            final_wealth = torch.tensor(evolved_metadata["Raw_Wealth"].values)
            final_gini = gini(final_wealth)
            
            # 4. Topological Balkanization check
            # We regenerate the topology based on the EVOLVED exposures to see if 
            # economic classes have structurally separated.
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
            
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G)
            modularity = nx.community.modularity(G, communities)
            
            return {"gini": final_gini, "modularity": modularity}
        return runner

    # Execute Sweep
    for inh in inheritance_sweep:
        results = run_monte_carlo(get_sim_runner(inh), n_seeds=n_seeds)
        mean_modularities.append(np.mean([r["modularity"] for r in results]))
        mean_ginis.append(np.mean([r["gini"] for r in results]))

    # Assertion 1: Inheritance drives Wealth Gini (Gradient Check)
    assert_monotonic_relationship(inheritance_sweep, mean_ginis, "positive")
    
    # Assertion 2: Wealth Gini drives Modularity (The "Balkanization Law")
    assert_monotonic_relationship(mean_ginis, mean_modularities, "positive")

    # Assertion 3: Statistical Significance between Extremes
    low_results = run_monte_carlo(get_sim_runner(inheritance_sweep[0]), n_seeds=n_seeds)
    high_results = run_monte_carlo(get_sim_runner(inheritance_sweep[-1]), n_seeds=n_seeds)
    
    high_mods = [r["modularity"] for r in high_results]
    low_mods = [r["modularity"] for r in low_results]
    assert_statistically_greater(high_mods, low_mods)
