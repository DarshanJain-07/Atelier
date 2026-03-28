import torch

from main import prepare_society_for_debug
from research_paper_tests._metrics import mean_edge_cosine_similarity
from research_paper_tests.config_schema import get_test_scenario


def test_network_topology_is_normalized_and_homophilous(tmp_path):
    scenario = get_test_scenario("network_topology")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "topology"), evolve=False
    )

    adjacency = society.adjacency_matrix
    assert adjacency is not None

    row_sums = torch.sparse.sum(adjacency, dim=1).to_dense()
    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=settings["row_sum_tolerance"],
    )

    edge_similarity = mean_edge_cosine_similarity(society.exposures, adjacency)
    assert edge_similarity > settings["min_edge_similarity"]
