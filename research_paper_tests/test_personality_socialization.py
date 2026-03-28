from main import prepare_society_for_debug
from research_paper_tests._metrics import average_neighbor_distance
from research_paper_tests.config_schema import get_test_scenario


def test_personality_socialization_reduces_neighbor_friction(tmp_path):
    base = get_test_scenario("personality_socialization_base").sim_config()
    socialized = get_test_scenario("personality_socialization_socialized").sim_config()

    base_society = prepare_society_for_debug(base, output_dir=str(tmp_path / "base"), evolve=False)
    socialized_society = prepare_society_for_debug(
        socialized, output_dir=str(tmp_path / "socialized"), evolve=False
    )

    assert average_neighbor_distance(
        socialized_society.personalities, socialized_society.adjacency_matrix
    ) < average_neighbor_distance(base_society.personalities, base_society.adjacency_matrix)
