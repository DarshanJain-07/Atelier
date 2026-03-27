import torch

from main import aggregate_social_state, create_sim_config


def test_virality_multiplier_stays_bounded_by_config():
    config = create_sim_config(num_agents=1000)
    influence = torch.ones(config.num_agents)

    consensus = torch.zeros(config.num_agents, 8)
    consensus[:, 0] = 1.0

    outliers = torch.zeros(config.num_agents, 8)
    outliers[:950, 0] = 0.2
    outliers[950:, 6] = 1.0
    outliers[950:, 5] = 1.0

    consensus_state = aggregate_social_state(
        config, consensus, influence, engagement_scores=torch.ones(config.num_agents)
    )
    outlier_state = aggregate_social_state(
        config,
        outliers,
        influence,
        engagement_scores=torch.cat([torch.ones(950), torch.full((50,), 2.0)]),
    )

    max_allowed = 1.0 + config.max_viral_multiplier
    assert consensus_state["max_outrage_multiplier"] <= max_allowed + 1e-3
    assert outlier_state["max_outrage_multiplier"] <= max_allowed + 1e-3
    assert consensus_state["mean_outrage_multiplier"] >= outlier_state["mean_outrage_multiplier"]
