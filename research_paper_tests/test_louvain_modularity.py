import community as community_louvain

from research_paper_tests._metrics import adjacency_to_graph
from research_paper_tests.config_schema import (
    get_test_scenario,
    prepare_scenario_society,
)


def test_homophily_raises_louvain_modularity(tmp_path):
    low_scenario = get_test_scenario("louvain_low")
    low = low_scenario.sim_config()
    high = get_test_scenario("louvain_high").sim_config()
    settings = low_scenario.settings()

    low_society = prepare_scenario_society(
        "louvain_low",
        tmp_path,
        enable_evolution=low.enable_evolution,
        output_name="low",
    )
    high_society = prepare_scenario_society(
        "louvain_high",
        tmp_path,
        enable_evolution=high.enable_evolution,
        output_name="high",
    )

    low_graph = adjacency_to_graph(low_society.adjacency_matrix)
    high_graph = adjacency_to_graph(high_society.adjacency_matrix)

    low_partition = community_louvain.best_partition(
        low_graph,
        random_state=settings["partition_seed"],
    )
    high_partition = community_louvain.best_partition(
        high_graph,
        random_state=settings["partition_seed"],
    )

    low_modularity = community_louvain.modularity(low_partition, low_graph)
    high_modularity = community_louvain.modularity(high_partition, high_graph)

    assert high_modularity > low_modularity
