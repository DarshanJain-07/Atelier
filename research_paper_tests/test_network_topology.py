import numpy as np
import torch

from research_paper_tests._metrics import mean_edge_topology_similarity
from research_paper_tests.config_schema import (
    DIMENSION_INDICES,
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater


def _mean_random_pair_similarity(
    config,
    exposures: torch.Tensor,
    personalities: torch.Tensor,
    pair_count: int,
) -> float:
    wealth_idx = DIMENSION_INDICES["Wealth"]
    topology_exposures = exposures.clone()
    topology_exposures[:, wealth_idx] = 0.0
    features = torch.cat([topology_exposures, personalities], dim=1)

    rng = np.random.default_rng(config.seed)
    left = rng.integers(0, config.num_agents, size=pair_count)
    right = rng.integers(0, config.num_agents - 1, size=pair_count)
    right = right + (right >= left)

    similarity = torch.nn.functional.cosine_similarity(
        features[left],
        features[right],
        dim=1,
    )
    return float(similarity.mean().item())


def test_network_topology_is_normalized_and_homophilous(tmp_path, n_seeds):
    scenario = get_test_scenario("network_topology")
    settings = scenario.settings()

    def run_topology():
        # Derive seed from current torch initial seed set by run_monte_carlo
        mc_seed = int(torch.initial_seed() % (2**32))
        
        society = prepare_scenario_society(
            "network_topology",
            tmp_path / f"topo_{mc_seed}",
            enable_evolution=False,
            output_name="topology",
            seed=mc_seed,
        )

        adjacency = society.adjacency_matrix
        assert adjacency is not None

        row_sums = torch.sparse.sum(adjacency, dim=1).to_dense()
        assert torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=settings["row_sum_tolerance"],
        )

        edge_similarity = mean_edge_topology_similarity(
            society.exposures,
            society.personalities,
            adjacency,
        )
        random_similarity = _mean_random_pair_similarity(
            society.config,
            society.exposures,
            society.personalities,
            pair_count=min(adjacency._nnz(), max(1024, society.config.num_agents * 8)),
        )
        return {
            "edge": edge_similarity,
            "random": random_similarity,
        }

    results = run_monte_carlo(run_topology, n_seeds=n_seeds)
    edge_dist = [r["edge"] for r in results]
    random_dist = [r["random"] for r in results]

    # Validation: Edge similarity in the constructed topology must be 
    # statistically greater than random pair similarity (homophily).
    assert_statistically_greater(edge_dist, random_dist)
