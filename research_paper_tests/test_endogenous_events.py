import torch
import numpy as np
from main import aggregate_social_state
from research_paper_tests.config_schema import (
    get_test_scenario,
    set_emotions,
    zero_emotions,
)
from research_paper_tests.stats_utils import run_monte_carlo

def test_endogenous_events_fire_only_for_unstable_societies(n_seeds):
    scenario = get_test_scenario("endogenous_events")
    config = scenario.sim_config()
    settings = scenario.settings()

    def runner():
        influence = torch.ones(config.num_agents)

        # Stable society setup
        stable = zero_emotions(config.num_agents)
        set_emotions(stable, settings["stable_emotion"])
        stable_state = aggregate_social_state(config, stable, influence)

        # Polarized society setup
        polarized = zero_emotions(config.num_agents)
        midpoint = config.num_agents // 2
        set_emotions(polarized, settings["polarized_group_a"], rows=slice(None, midpoint))
        set_emotions(polarized, settings["polarized_group_b"], rows=slice(midpoint, None))
        polarized_state = aggregate_social_state(config, polarized, influence)
        
        return (
            stable_state.get("action_name") is None,
            polarized_state.get("action_name") in settings["allowed_actions"]
        )

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    
    # Validation: Events should NEVER fire in stable societies and ALWAYS fire in polarized ones
    assert all(r[0] for r in results), "Event fired in a stable society"
    assert all(r[1] for r in results), "Event failed to fire in a polarized society"
