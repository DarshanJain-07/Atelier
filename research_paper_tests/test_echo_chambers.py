from main import prepare_society_for_debug
from research_paper_tests._metrics import mean_edge_cosine_similarity
from research_paper_tests.config_schema import get_test_scenario


def test_homophilous_topology_forms_stronger_echo_chambers(tmp_path):
    low_homophily = get_test_scenario("echo_chambers_low").sim_config()
    high_homophily = get_test_scenario("echo_chambers_high").sim_config()

    low_society = prepare_society_for_debug(
        low_homophily, output_dir=str(tmp_path / "low"), evolve=False
    )
    high_society = prepare_society_for_debug(
        high_homophily, output_dir=str(tmp_path / "high"), evolve=False
    )

    low_similarity = mean_edge_cosine_similarity(
        low_society.exposures, low_society.adjacency_matrix
    )
    high_similarity = mean_edge_cosine_similarity(
        high_society.exposures, high_society.adjacency_matrix
    )

    assert high_similarity > low_similarity
