import torch
import numpy as np
from main import aggregate_social_state
from research_paper_tests.config_schema import (
    get_test_scenario,
    set_emotions,
    zero_emotions,
)
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater

def run_virality_simulation(is_outlier_scenario=False):
    scenario = get_test_scenario("maximum_virality")
    config = scenario.sim_config()
    settings = scenario.settings()
    influence = torch.ones(config.num_agents)
    
    if not is_outlier_scenario:
        # Consensus state
        emotions = zero_emotions(config.num_agents)
        set_emotions(emotions, settings["consensus_emotion"])
        engagement = torch.ones(config.num_agents)
    else:
        # Outlier state
        emotions = zero_emotions(config.num_agents)
        mainstream_count = settings["outlier_mainstream_count"]
        set_emotions(
            emotions,
            settings["outlier_mainstream_emotion"],
            rows=slice(None, mainstream_count),
        )
        set_emotions(emotions, settings["outlier_emotions"], rows=slice(mainstream_count, None))
        engagement = torch.cat(
            [
                torch.ones(mainstream_count),
                torch.full((settings["outlier_count"],), settings["boosted_engagement"]),
            ],
        )

    state = aggregate_social_state(
        config, emotions, influence, engagement_scores=engagement,
    )
    return state

def test_virality_multiplier_statistical():
    """
    Goal: Statistically verify that virality multipliers stay bounded and 
    behave correctly in consensus vs outlier scenarios.
    """
    print("\nRunning Monte Carlo for Maximum Virality...")
    
    consensus_results = run_monte_carlo(lambda: run_virality_simulation(is_outlier_scenario=False))
    outlier_results = run_monte_carlo(lambda: run_virality_simulation(is_outlier_scenario=True))
    
    consensus_max = [r["max_outrage_multiplier"] for r in consensus_results]
    outlier_max = [r["max_outrage_multiplier"] for r in outlier_results]
    
    consensus_mean_outrage = [r["mean_outrage_multiplier"] for r in consensus_results]
    outlier_mean_outrage = [r["mean_outrage_multiplier"] for r in outlier_results]

    scenario = get_test_scenario("maximum_virality")
    config = scenario.sim_config()
    settings = scenario.settings()
    max_allowed = 1.0 + config.max_viral_multiplier
    tolerance = settings["max_multiplier_tolerance"]
    
    print(f"Consensus Max Multiplier: {np.mean(consensus_max):.3f}")
    print(f"Outlier Max Multiplier: {np.mean(outlier_max):.3f}")
    
    # Assertions
    assert np.mean(consensus_max) <= max_allowed + tolerance
    assert np.mean(outlier_max) <= max_allowed + tolerance
    
    # In consensus, the mean outrage multiplier should be higher (more distributed contagion)
    # whereas in outlier it might be concentrated. 
    # (Based on original test logic: assert consensus_state["mean_outrage_multiplier"] >= outlier_state["mean_outrage_multiplier"])
    assert_statistically_greater(consensus_mean_outrage, outlier_mean_outrage)

if __name__ == "__main__":
    test_virality_multiplier_statistical()
