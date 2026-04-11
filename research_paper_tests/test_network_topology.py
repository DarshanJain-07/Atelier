import numpy as np
import torch

from research_paper_tests._metrics import mean_edge_topology_similarity
from research_paper_tests.config_schema import (
    get_test_scenario,
    prepare_scenario_society,
)
from schema import DIMENSION_INDICES


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


def test_network_topology_is_normalized_and_homophilous(tmp_path):
    scenario = get_test_scenario("network_topology")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "network_topology",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="topology",
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
        config,
        society.exposures,
        society.personalities,
        pair_count=min(adjacency._nnz(), max(1024, config.num_agents * 8)),
    )
    assert edge_similarity > random_similarity
