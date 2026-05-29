import numpy as np
import torch
from research_paper_tests._metrics import mean_edge_topology_similarity
from research_paper_tests.config_schema import prepare_scenario_society
from research_paper_tests.stats_utils import (
    run_monte_carlo,
    assert_monotonic_relationship,
    assert_statistically_greater
)

def test_homophilous_topology_forms_stronger_echo_chambers(tmp_path, n_seeds):
    """
    Validation: Higher homophily strength must lead to higher edge similarity.
    This replaces the naive high vs low check with a gradient-based statistical proof.
    """
    homophily_sweep = [1.0, 4.0, 8.0, 12.0]
    mean_similarities = []

    def get_sim_runner(h_val):
        def runner():
            # Use a smaller population for the sweep to keep tests fast
            society = prepare_scenario_society(
                "echo_chambers_low",
                tmp_path / f"h_{h_val}_{np.random.randint(1e6)}",
                enable_evolution=False,
                homophily_strength=h_val,
                num_agents=100, 
            )
            return mean_edge_topology_similarity(
                society.exposures,
                society.personalities,
                society.adjacency_matrix,
            )
        return runner

    # 1. Gradient Sweep
    for h in homophily_sweep:
        results = run_monte_carlo(get_sim_runner(h), n_seeds=n_seeds)
        mean_similarities.append(np.mean(results))

    # Assert that as homophily increases, similarity strictly increases (Spearman > 0, p < 0.05)
    assert_monotonic_relationship(homophily_sweep, mean_similarities, "positive")

    # 2. Extreme Significance Check
    # Verify that the highest homophily is statistically distinct from the lowest
    low_results = run_monte_carlo(get_sim_runner(homophily_sweep[0]), n_seeds=n_seeds)
    high_results = run_monte_carlo(get_sim_runner(homophily_sweep[-1]), n_seeds=n_seeds)
    assert_statistically_greater(high_results, low_results)
