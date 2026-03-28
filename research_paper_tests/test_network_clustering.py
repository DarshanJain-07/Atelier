import numpy as np
import torch

from main import apply_triadic_closure_for_debug, create_topology_for_debug
from research_paper_tests._metrics import average_clustering
from research_paper_tests.config_schema import (
    PERSONALITY_TRAIT_COUNT,
    WORLD_DIMENSION_COUNT,
    get_test_scenario,
)


def test_triadic_closure_increases_average_clustering():
    scenario = get_test_scenario("network_clustering_closure")
    backbone_config = get_test_scenario("network_clustering_backbone").sim_config()
    closure_config = scenario.sim_config()
    settings = scenario.settings()

    torch.manual_seed(settings["torch_seed"])
    np.random.seed(settings["numpy_seed"])
    exposures = torch.randn(backbone_config.num_agents, WORLD_DIMENSION_COUNT)
    personalities = torch.sigmoid(
        torch.randn(backbone_config.num_agents, PERSONALITY_TRAIT_COUNT)
    )
    influence = np.random.lognormal(
        mean=settings["influence_mean"],
        sigma=settings["influence_std"],
        size=backbone_config.num_agents,
    )

    backbone = create_topology_for_debug(
        backbone_config, exposures, personalities, influence
    )
    refined = apply_triadic_closure_for_debug(closure_config, backbone)

    assert average_clustering(refined) >= average_clustering(backbone)
