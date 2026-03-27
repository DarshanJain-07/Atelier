import torch

from main import aggregate_social_state, create_sim_config


def test_endogenous_events_fire_only_for_unstable_societies():
    config = create_sim_config(
        num_agents=1000,
        elite_divergence_threshold=0.3,
        polarization_threshold=0.3,
        stewing_ticks=1,
    )

    influence = torch.ones(config.num_agents)

    stable = torch.zeros(config.num_agents, 8)
    stable[:, 0] = 0.5
    stable_state = aggregate_social_state(config, stable, influence)

    polarized = torch.zeros(config.num_agents, 8)
    polarized[:500, 6] = 1.0
    polarized[500:, 0] = 1.0
    polarized_state = aggregate_social_state(config, polarized, influence)

    assert stable_state.get("action_name") is None
    assert polarized_state.get("action_name") in {
        "Civil Protest",
        "Populist Uprising",
        "Elite Policy Shift",
    }
