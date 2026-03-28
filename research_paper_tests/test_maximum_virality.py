import torch

from main import aggregate_social_state
from research_paper_tests.config_schema import (
    get_test_scenario,
    set_emotions,
    zero_emotions,
)


def test_virality_multiplier_stays_bounded_by_config():
    scenario = get_test_scenario("maximum_virality")
    config = scenario.sim_config()
    settings = scenario.settings()
    influence = torch.ones(config.num_agents)

    consensus = zero_emotions(config.num_agents)
    set_emotions(consensus, settings["consensus_emotion"])

    outliers = zero_emotions(config.num_agents)
    mainstream_count = settings["outlier_mainstream_count"]
    set_emotions(
        outliers,
        settings["outlier_mainstream_emotion"],
        rows=slice(None, mainstream_count),
    )
    set_emotions(outliers, settings["outlier_emotions"], rows=slice(mainstream_count, None))

    consensus_state = aggregate_social_state(
        config, consensus, influence, engagement_scores=torch.ones(config.num_agents)
    )
    outlier_state = aggregate_social_state(
        config,
        outliers,
        influence,
        engagement_scores=torch.cat(
            [
                torch.ones(mainstream_count),
                torch.full((settings["outlier_count"],), settings["boosted_engagement"]),
            ]
        ),
    )

    max_allowed = 1.0 + config.max_viral_multiplier
    assert (
        consensus_state["max_outrage_multiplier"]
        <= max_allowed + settings["max_multiplier_tolerance"]
    )
    assert (
        outlier_state["max_outrage_multiplier"]
        <= max_allowed + settings["max_multiplier_tolerance"]
    )
    assert consensus_state["mean_outrage_multiplier"] >= outlier_state["mean_outrage_multiplier"]
