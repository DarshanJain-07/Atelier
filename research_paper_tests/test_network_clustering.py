import numpy as np
import torch

from main import apply_triadic_closure_for_debug, create_sim_config, create_topology_for_debug
from research_paper_tests._metrics import average_clustering


def test_triadic_closure_increases_average_clustering():
    config = create_sim_config(
        num_agents=400,
        base_connections=10,
        use_network_topology=True,
        enable_evolution=False,
    )

    torch.manual_seed(42)
    np.random.seed(42)
    exposures = torch.randn(config.num_agents, 12)
    personalities = torch.sigmoid(torch.randn(config.num_agents, 5))
    influence = np.random.lognormal(mean=1.0, sigma=0.5, size=config.num_agents)

    backbone_config = create_sim_config(
        num_agents=config.num_agents,
        base_connections=10,
        triadic_closure_prob=0.0,
        use_network_topology=True,
        enable_evolution=False,
    )
    closure_config = create_sim_config(
        num_agents=config.num_agents,
        base_connections=10,
        triadic_closure_prob=0.3,
        use_network_topology=True,
        enable_evolution=False,
    )

    backbone = create_topology_for_debug(
        backbone_config, exposures, personalities, influence
    )
    refined = apply_triadic_closure_for_debug(closure_config, backbone)

    assert average_clustering(refined) >= average_clustering(backbone)
