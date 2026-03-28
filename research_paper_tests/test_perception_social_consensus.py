import torch

from main import distort_world_signal, prepare_society_for_debug
from research_paper_tests._metrics import average_neighbor_distance
from research_paper_tests.config_schema import build_world, get_test_scenario


def test_social_consensus_aligns_neighbor_perceptions(tmp_path):
    scenario = get_test_scenario("perception_social_consensus")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "consensus"), evolve=False
    )

    world = build_world(settings["world"])

    baseline_config = get_test_scenario("perception_social_consensus_baseline").sim_config(
        num_agents=config.num_agents
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
