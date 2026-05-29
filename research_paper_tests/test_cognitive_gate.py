import numpy as np
import torch
from main import build_debug_society, run_debug_simulation
from research_paper_tests.config_schema import (
    DIMENSION_INDICES,
    PERSONALITY_INDICES,
    WORLD_DIMENSION_COUNT,
    build_world,
    get_test_scenario,
    zero_personalities,
)
from research_paper_tests.stats_utils import (
    run_monte_carlo,
    assert_monotonic_relationship,
    assert_statistically_greater
)

def test_cognitive_gate_gradient_response(n_seeds):
    """
    Validation: Engagement must increase monotonically with Openness for misaligned signals.
    High-Openness agents "gate" less, allowing more engagement even with semantic contradiction.
    This replaces hardcoded threshold probes with a gradient-based behavioral law.
    """
    scenario = get_test_scenario("figure_cognitive_gate")
    settings = scenario.settings()
    
    # We sweep Openness to prove the "Gate" opens linearly or monotonically
    openness_sweep = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    mean_engagements = []

    def get_sim_runner(openness_val):
        def runner():
            # Build a focused test society for this specific openness level
            config = scenario.sim_config()
            config.num_agents = 50 
            
            exposures = torch.zeros(config.num_agents, WORLD_DIMENSION_COUNT)
            personalities = zero_personalities(config.num_agents, fill=settings["trait_fill"])
            personalities[:, PERSONALITY_INDICES["Openness"]] = openness_val
            
            # Setup strict misalignment: world signal is Positive, agent exposure is Negative
            for dimension_name, dimension_value in settings["aligned_worldview"].items():
                exposures[:, DIMENSION_INDICES[dimension_name]] = -1.0 
            
            society = build_debug_society(config, exposures, personalities)
            world = build_world(settings["world"]) # The signal is Positive
            
            result = run_debug_simulation(
                config,
                world,
                society=society,
                urgency=settings["urgency"],
            )
            return result.engagement_scores.mean().item()
        return runner

    # 1. Execute Sweep
    for o in openness_sweep:
        results = run_monte_carlo(get_sim_runner(o), n_seeds=n_seeds)
        mean_engagements.append(np.mean(results))

    # Assertion: Increasing Openness must increase Engagement (lowers the "Gate" resistance)
    assert_monotonic_relationship(openness_sweep, mean_engagements, "positive")

    # 2. Statistical Significance
    low_open_results = run_monte_carlo(get_sim_runner(0.0), n_seeds=n_seeds)
    high_open_results = run_monte_carlo(get_sim_runner(1.0), n_seeds=n_seeds)
    assert_statistically_greater(high_open_results, low_open_results)
