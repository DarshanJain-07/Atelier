from main import create_sim_config, prepare_society_for_debug
from research_paper_tests._metrics import mean_edge_cosine_similarity


def test_homophilous_topology_forms_stronger_echo_chambers(tmp_path):
    low_homophily = create_sim_config(
        num_agents=500,
        homophily_strength=0.1,  # Almost random
        influence_bias_exp=1.0,  # High influence bridging
        use_network_topology=True,
        personality_socialization_gain=0.0,
        enable_evolution=False,
    )
    high_homophily = create_sim_config(
        num_agents=500,
        homophily_strength=8.0,  # Aggressive homophily
        influence_bias_exp=0.1,  # Low influence bridging
        use_network_topology=True,
        personality_socialization_gain=0.0,
        enable_evolution=False,
    )

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
