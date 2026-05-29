import torch
from main import distort_world_signal
from research_paper_tests._metrics import average_neighbor_distance
from research_paper_tests.config_schema import (
    build_world,
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater


def test_social_consensus_aligns_neighbor_perceptions(tmp_path, n_seeds):
    scenario = get_test_scenario("perception_social_consensus")
    config = scenario.sim_config()
    settings = scenario.settings()
    
    baseline_scenario = get_test_scenario("perception_social_consensus_baseline")

    def runner():
        society = prepare_scenario_society(
            "perception_social_consensus",
            tmp_path,
            enable_evolution=config.enable_evolution,
            output_name="consensus",
        )

        world = build_world(settings["world"])

        baseline_config = baseline_scenario.sim_config(
            num_agents=config.num_agents,
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

        return {
            "baseline_dist": average_neighbor_distance(baseline, society.adjacency_matrix),
            "consensus_dist": average_neighbor_distance(consensus, society.adjacency_matrix),
        }

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    baseline_dists = [r["baseline_dist"] for r in results]
    consensus_dists = [r["consensus_dist"] for r in results]

    assert_statistically_greater(baseline_dists, consensus_dists)
