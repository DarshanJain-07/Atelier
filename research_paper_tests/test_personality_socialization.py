from main import create_sim_config, prepare_society_for_debug
from research_paper_tests._metrics import average_neighbor_distance


def test_personality_socialization_reduces_neighbor_friction(tmp_path):
    base = create_sim_config(
        num_agents=500,
        use_network_topology=True,
        personality_socialization_gain=0.0,
        enable_evolution=False,
    )
    socialized = create_sim_config(
        num_agents=500,
        use_network_topology=True,
        personality_socialization_gain=0.4,
        enable_evolution=False,
    )

    base_society = prepare_society_for_debug(base, output_dir=str(tmp_path / "base"), evolve=False)
    socialized_society = prepare_society_for_debug(
        socialized, output_dir=str(tmp_path / "socialized"), evolve=False
    )

    assert average_neighbor_distance(
        socialized_society.personalities, socialized_society.adjacency_matrix
    ) < average_neighbor_distance(base_society.personalities, base_society.adjacency_matrix)
