from research_paper_tests._metrics import average_neighbor_distance
from research_paper_tests.config_schema import (
    get_test_scenario,
    prepare_scenario_society,
)


def test_personality_socialization_reduces_neighbor_friction(tmp_path):
    base = get_test_scenario("personality_socialization_base").sim_config()
    socialized = get_test_scenario("personality_socialization_socialized").sim_config()

    base_society = prepare_scenario_society(
        "personality_socialization_base",
        tmp_path,
        enable_evolution=base.enable_evolution,
        output_name="base",
    )
    socialized_society = prepare_scenario_society(
        "personality_socialization_socialized",
        tmp_path,
        enable_evolution=socialized.enable_evolution,
        output_name="socialized",
    )

    assert average_neighbor_distance(
        socialized_society.personalities, socialized_society.adjacency_matrix
    ) < average_neighbor_distance(base_society.personalities, base_society.adjacency_matrix)
