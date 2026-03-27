import torch

from main import DIMENSION_INDICES, build_debug_society, create_sim_config, run_debug_simulation


def test_relative_deprivation_hits_marginalized_agents_harder():
    config = create_sim_config(
        num_agents=200,
        use_signal_distortion=False,
        use_time_pressure=False,
        use_network_topology=False,
        enable_evolution=False,
    )

    exposures_marginalized = torch.zeros(100, 12)
    exposures_marginalized[:, DIMENSION_INDICES["Wealth"]] = -0.8
    exposures_marginalized[:, DIMENSION_INDICES["Fairness"]] = -0.8

    personalities_marginalized = torch.ones(100, 5) * 0.5
    personalities_marginalized[:, 3] = 0.1
    personalities_marginalized[:, 4] = 0.9

    exposures_elites = torch.zeros(100, 12)
    exposures_elites[:, DIMENSION_INDICES["Wealth"]] = 0.8
    exposures_elites[:, DIMENSION_INDICES["Fairness"]] = 0.8

    personalities_elites = torch.ones(100, 5) * 0.5
    personalities_elites[:, 3] = 0.9
    personalities_elites[:, 4] = 0.1

    society = build_debug_society(
        config,
        torch.cat([exposures_marginalized, exposures_elites], dim=0),
        torch.cat([personalities_marginalized, personalities_elites], dim=0),
    )

    world = torch.zeros(1, 12)
    world[0, DIMENSION_INDICES["Wealth"]] = 0.5
    world[0, DIMENSION_INDICES["Fairness"]] = -1.0

    result = run_debug_simulation(config, world, society=society, urgency=0.0)
    anger = result.final_emotions[:, 6]

    assert anger[:100].mean().item() > anger[100:].mean().item()
