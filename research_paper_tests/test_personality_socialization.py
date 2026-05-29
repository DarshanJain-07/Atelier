import torch
from research_paper_tests._metrics import average_neighbor_distance
from research_paper_tests.config_schema import (
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.stats_utils import assert_statistically_greater, run_monte_carlo


def test_personality_socialization_reduces_neighbor_friction(tmp_path):
    def run_case(scenario_name):
        scenario = get_test_scenario(scenario_name)
        config = scenario.sim_config()
        # Use a unique output name per seed to avoid collisions in tmp_path
        seed = torch.initial_seed()
        society = prepare_scenario_society(
            scenario_name,
            tmp_path,
            enable_evolution=config.enable_evolution,
            output_name=f"{scenario_name}_{seed}",
        )
        return average_neighbor_distance(
            society.personalities,
            society.adjacency_matrix,
        )

    base_dist = run_monte_carlo(lambda: run_case("personality_socialization_base"))
    socialized_dist = run_monte_carlo(
        lambda: run_case("personality_socialization_socialized"),
    )

    # We expect socialized to have LOWER distance, so base > socialized
    assert_statistically_greater(base_dist, socialized_dist)
