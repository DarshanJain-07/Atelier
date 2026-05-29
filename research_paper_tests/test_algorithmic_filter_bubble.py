import torch
import numpy as np
from main import (
    ALGORITHMIC_DENSE_REWEIGHT_MAX_AGENTS,
    reweight_algorithmic_adjacency,
    run_debug_simulation,
)
from research_paper_tests.config_schema import (
    build_world,
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.stats_utils import (
    run_monte_carlo,
    assert_statistically_greater,
    assert_monotonic_relationship
)


def _two_cluster_sparse_adjacency(num_agents: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert num_agents % 2 == 0
    half = num_agents // 2
    exposures = torch.zeros(num_agents, 12)
    exposures[:half, 0] = 1.0
    exposures[half:, 0] = -1.0

    rows = []
    cols = []
    for agent_idx in range(num_agents):
        if agent_idx < half:
            same_neighbor = (agent_idx + 1) % half
            cross_neighbor = half + agent_idx
        else:
            same_neighbor = half + ((agent_idx - half + 1) % half)
            cross_neighbor = agent_idx - half
        rows.extend([agent_idx, agent_idx])
        cols.extend([same_neighbor, cross_neighbor])

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.full((len(rows),), 0.5, dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_agents, num_agents),
    ).coalesce()
    return adjacency, exposures


def _same_and_cross_edge_means(
    adjacency: torch.Tensor,
    num_agents: int,
) -> tuple[float, float]:
    adjacency = adjacency.coalesce()
    half = num_agents // 2
    rows = adjacency.indices()[0]
    cols = adjacency.indices()[1]
    values = adjacency.values()
    same_cluster = (rows < half) == (cols < half)
    return (
        float(values[same_cluster].mean().item()),
        float(values[~same_cluster].mean().item()),
    )


def test_algorithmic_reweighting_uses_dense_small_population(n_seeds):
    def runner(seed: int):
        torch.manual_seed(seed)
        num_agents = min(256, ALGORITHMIC_DENSE_REWEIGHT_MAX_AGENTS)
        adjacency, exposures = _two_cluster_sparse_adjacency(num_agents)

        reweighted, mode = reweight_algorithmic_adjacency(
            adjacency,
            exposures,
            torch.tensor([0]),
        )

        row_sums = torch.sparse.sum(reweighted, dim=1).to_dense()
        same_mean, cross_mean = _same_and_cross_edge_means(reweighted, num_agents)
        return {
            "mode": mode,
            "row_sum_error": float((row_sums - 1.0).abs().max().item()),
            "same_mean": same_mean,
            "cross_mean": cross_mean,
        }

    results = run_monte_carlo(runner, n_seeds=min(n_seeds, 3))
    assert {result["mode"] for result in results} == {"dense"}
    assert max(result["row_sum_error"] for result in results) < 1e-6
    assert_statistically_greater(
        [result["same_mean"] for result in results],
        [result["cross_mean"] for result in results],
    )


def test_algorithmic_reweighting_uses_sparse_large_population(n_seeds):
    def runner(seed: int):
        torch.manual_seed(seed)
        num_agents = 1200
        adjacency, exposures = _two_cluster_sparse_adjacency(num_agents)

        reweighted, mode = reweight_algorithmic_adjacency(
            adjacency,
            exposures,
            torch.tensor([0]),
        )

        row_sums = torch.sparse.sum(reweighted, dim=1).to_dense()
        same_mean, cross_mean = _same_and_cross_edge_means(reweighted, num_agents)
        return {
            "mode": mode,
            "nnz": reweighted._nnz(),
            "row_sum_error": float((row_sums - 1.0).abs().max().item()),
            "same_mean": same_mean,
            "cross_mean": cross_mean,
        }

    results = run_monte_carlo(runner, n_seeds=min(n_seeds, 3))
    assert {result["mode"] for result in results} == {"sparse"}
    assert {result["nnz"] for result in results} == {2400}
    assert max(result["row_sum_error"] for result in results) < 1e-6
    assert_statistically_greater(
        [result["same_mean"] for result in results],
        [result["cross_mean"] for result in results],
    )

def test_algorithmic_amplification_intensity_gradient(tmp_path, n_seeds):
    """
    Validation: Increasing the algorithmic exaggeration factor must monotonically 
    increase the mean engagement scores across the society.
    This replaces the naive high/low check with a proof of algorithmic scaling laws.
    """
    scenario = get_test_scenario("algorithmic_filter_bubble")
    settings = scenario.settings()
    
    # Sweep the exaggeration factor to prove the "Filter Bubble" intensity scales
    exaggeration_sweep = [1.0, 1.5, 2.0, 3.0, 4.0]
    mean_engagements = []

    def get_sim_runner(exaggeration):
        def runner():
            # Build society with specific amplification intensity
            society = prepare_scenario_society(
                "algorithmic_filter_bubble",
                tmp_path / f"algo_{exaggeration}_{np.random.randint(1e6)}",
                enable_evolution=False,
                use_algorithmic_amplification=True,
                algo_exaggeration_factor=exaggeration,
                num_agents=200, # Scaled for faster iteration
            )
            world = build_world(settings["world"])
            
            result = run_debug_simulation(
                society.config,
                world,
                society=society,
                urgency=settings["urgency"],
            )
            return result.engagement_scores.mean().item()
        return runner

    # 1. Gradient Sweep
    for ex in exaggeration_sweep:
        results = run_monte_carlo(get_sim_runner(ex), n_seeds=n_seeds)
        mean_engagements.append(np.mean(results))

    # Assertion: As the algorithm becomes more aggressive, engagement must increase
    assert_monotonic_relationship(exaggeration_sweep, mean_engagements, "positive")

    # 2. Statistical Significance: Neutral (1.0) vs Highly Amplified (4.0)
    neutral_results = run_monte_carlo(get_sim_runner(1.0), n_seeds=n_seeds)
    amplified_results = run_monte_carlo(get_sim_runner(4.0), n_seeds=n_seeds)
    
    # Validation: High amplification must create a statistically distinct outcome 
    # compared to no amplification.
    assert_statistically_greater(amplified_results, neutral_results)
