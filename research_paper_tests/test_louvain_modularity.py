import community as community_louvain

from main import create_sim_config, prepare_society_for_debug
from research_paper_tests._metrics import adjacency_to_graph


def test_homophily_raises_louvain_modularity(tmp_path):
    low = create_sim_config(
        num_agents=500,
        homophily_strength=1.0,
        use_network_topology=True,
        enable_evolution=False,
    )
    high = create_sim_config(
        num_agents=500,
        homophily_strength=8.0,
        use_network_topology=True,
        enable_evolution=False,
        influence_bias_exp=0.0, # Stop influencers from bridging clusters
        base_connections=2,     # Lower density prevents giant component blob
        triadic_closure_prob=0.8, # Tighten existing clusters
    )

    low_society = prepare_society_for_debug(low, output_dir=str(tmp_path / "low"), evolve=False)
    high_society = prepare_society_for_debug(high, output_dir=str(tmp_path / "high"), evolve=False)

    low_graph = adjacency_to_graph(low_society.adjacency_matrix)
    high_graph = adjacency_to_graph(high_society.adjacency_matrix)

    low_partition = community_louvain.best_partition(low_graph)
    high_partition = community_louvain.best_partition(high_graph)

    low_modularity = community_louvain.modularity(low_partition, low_graph)
    high_modularity = community_louvain.modularity(high_partition, high_graph)

    assert high_modularity > low_modularity
