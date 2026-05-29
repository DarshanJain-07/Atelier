import numpy as np
import torch

from main import apply_triadic_closure_for_debug, create_topology_for_debug
from research_paper_tests._metrics import average_clustering
from research_paper_tests.config_schema import (
    PERSONALITY_TRAIT_COUNT,
    WORLD_DIMENSION_COUNT,
    get_test_scenario,
)
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater


def test_triadic_closure_increases_average_clustering(n_seeds):
    scenario = get_test_scenario("network_clustering_closure")
    backbone_config = get_test_scenario("network_clustering_backbone").sim_config()
    closure_config = scenario.sim_config()
    settings = scenario.settings()

    def run_clustering():
        # run_monte_carlo handles seeding for torch and numpy internally.
        exposures = torch.randn(backbone_config.num_agents, WORLD_DIMENSION_COUNT)
        personalities = torch.sigmoid(
            torch.randn(backbone_config.num_agents, PERSONALITY_TRAIT_COUNT),
        )
        influence = np.random.lognormal(
            mean=settings["influence_mean"],
            sigma=settings["influence_std"],
            size=backbone_config.num_agents,
        )

        backbone = create_topology_for_debug(
            backbone_config, exposures, personalities, influence,
        )
        refined = apply_triadic_closure_for_debug(closure_config, backbone)

        return {
            "backbone": average_clustering(backbone),
            "refined": average_clustering(refined),
        }

    results = run_monte_carlo(run_clustering, n_seeds=n_seeds)
    backbone_dist = [r["backbone"] for r in results]
    refined_dist = [r["refined"] for r in results]

    # Validation: Triadic closure must statistically increase the clustering coefficient
    # compared to the backbone topology.
    assert_statistically_greater(refined_dist, backbone_dist)
