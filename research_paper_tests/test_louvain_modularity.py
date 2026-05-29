import community as community_louvain
import numpy as np
from research_paper_tests._metrics import adjacency_to_graph
from research_paper_tests.config_schema import prepare_scenario_society
from research_paper_tests.stats_utils import (
    run_monte_carlo,
    assert_monotonic_relationship,
    assert_statistically_greater
)

def test_homophily_raises_louvain_modularity(tmp_path, n_seeds):
    """
    Validation: Structural modularity (community detection) must increase monotonically 
    with homophily strength. This proves that agents successfully self-segregate.
    """
    homophily_sweep = [1.0, 4.0, 8.0, 12.0]
    mean_modularities = []

    def get_sim_runner(h_val):
        def runner():
            society = prepare_scenario_society(
                "louvain_low",
                tmp_path / f"louvain_{h_val}_{np.random.randint(1e6)}",
                enable_evolution=False,
                homophily_strength=h_val,
                num_agents=150,
            )
            graph = adjacency_to_graph(society.adjacency_matrix)
            # Louvain partition can be stochastic, so we measure its output across seeds
            partition = community_louvain.best_partition(graph)
            return community_louvain.modularity(partition, graph)
        return runner

    # 1. Gradient Sweep
    for h in homophily_sweep:
        results = run_monte_carlo(get_sim_runner(h), n_seeds=n_seeds)
        mean_modularities.append(np.mean(results))

    assert_monotonic_relationship(homophily_sweep, mean_modularities, "positive")

    # 2. Statistical Significance between extreme states
    low_results = run_monte_carlo(get_sim_runner(homophily_sweep[0]), n_seeds=n_seeds)
    high_results = run_monte_carlo(get_sim_runner(homophily_sweep[-1]), n_seeds=n_seeds)
    assert_statistically_greater(high_results, low_results)
