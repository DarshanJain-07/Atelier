import torch

from main import create_sim_config, prepare_society_for_debug
from research_paper_tests._metrics import mean_edge_cosine_similarity


def test_network_topology_is_normalized_and_homophilous(tmp_path):
    config = create_sim_config(
        num_agents=400,
        use_network_topology=True,
        base_connections=20,
        homophily_strength=3.0,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "topology"), evolve=False
    )

    adjacency = society.adjacency_matrix
    assert adjacency is not None

    row_sums = torch.sparse.sum(adjacency, dim=1).to_dense()
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    edge_similarity = mean_edge_cosine_similarity(society.exposures, adjacency)
    assert edge_similarity > 0.05
