import torch

from main import aggregate_social_state
from research_paper_tests.config_schema import (
    get_test_scenario,
    set_emotions,
    zero_emotions,
)


def test_endogenous_events_fire_only_for_unstable_societies():
    scenario = get_test_scenario("endogenous_events")
    config = scenario.sim_config()
    settings = scenario.settings()

    influence = torch.ones(config.num_agents)

    stable = zero_emotions(config.num_agents)
    set_emotions(stable, settings["stable_emotion"])
    stable_state = aggregate_social_state(config, stable, influence)

    polarized = zero_emotions(config.num_agents)
    midpoint = config.num_agents // 2
    set_emotions(polarized, settings["polarized_group_a"], rows=slice(None, midpoint))
    set_emotions(polarized, settings["polarized_group_b"], rows=slice(midpoint, None))
    polarized_state = aggregate_social_state(config, polarized, influence)

    assert stable_state.get("action_name") is None
    assert polarized_state.get("action_name") in settings["allowed_actions"]
