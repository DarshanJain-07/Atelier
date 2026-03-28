from research_paper_tests._metrics import mean_edge_cosine_similarity
from research_paper_tests.config_schema import (
    get_test_scenario,
    prepare_scenario_society,
)


def test_homophilous_topology_forms_stronger_echo_chambers(tmp_path):
    low_homophily = get_test_scenario("echo_chambers_low").sim_config()
    high_homophily = get_test_scenario("echo_chambers_high").sim_config()

    low_society = prepare_scenario_society(
        "echo_chambers_low",
        tmp_path,
        enable_evolution=low_homophily.enable_evolution,
        output_name="low",
    )
    high_society = prepare_scenario_society(
        "echo_chambers_high",
        tmp_path,
        enable_evolution=high_homophily.enable_evolution,
        output_name="high",
    )

    low_similarity = mean_edge_cosine_similarity(
        low_society.exposures, low_society.adjacency_matrix
    )
    high_similarity = mean_edge_cosine_similarity(
        high_society.exposures, high_society.adjacency_matrix
    )

    assert high_similarity > low_similarity
