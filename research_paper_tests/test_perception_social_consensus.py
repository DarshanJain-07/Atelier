import torch

from main import DIMENSION_INDICES, create_sim_config, distort_world_signal, prepare_society_for_debug
from research_paper_tests._metrics import average_neighbor_distance


def test_social_consensus_aligns_neighbor_perceptions(tmp_path):
    config = create_sim_config(
        num_agents=500,
        use_signal_distortion=True,
        distortion_max_noise=0.6,
        distortion_neurotic_gain=1.0,
        use_network_topology=True,
        perception_social_consensus_gain=0.3,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "consensus"), evolve=False
    )

    world = torch.zeros(1, 12)
    world[0, DIMENSION_INDICES["Physical_Safety"]] = -0.5

    baseline_config = create_sim_config(
        num_agents=config.num_agents,
        use_signal_distortion=True,
        distortion_max_noise=0.6,
        distortion_neurotic_gain=1.0,
        use_network_topology=True,
        perception_social_consensus_gain=0.0,
        enable_evolution=False,
    )
    baseline = distort_world_signal(
        baseline_config,
        world,
        society.personalities,
        adjacency_matrix=society.adjacency_matrix,
    )
    consensus = (1.0 - config.perception_social_consensus_gain) * baseline + (
        config.perception_social_consensus_gain
        * torch.sparse.mm(society.adjacency_matrix, baseline)
    )

    assert average_neighbor_distance(consensus, society.adjacency_matrix) < average_neighbor_distance(
        baseline, society.adjacency_matrix
    )
