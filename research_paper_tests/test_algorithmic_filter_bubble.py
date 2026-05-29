import torch
import numpy as np
from main import run_debug_simulation
from research_paper_tests.config_schema import (
    build_world,
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.stats_utils import (
    run_monte_carlo,
    assert_statistically_greater,
    assert_monotonic_relationship
)

def test_algorithmic_amplification_intensity_gradient(tmp_path, n_seeds):
    """
    Validation: Increasing the algorithmic exaggeration factor must monotonically 
    increase the mean engagement scores across the society.
    This replaces the naive high/low check with a proof of algorithmic scaling laws.
    """
    scenario = get_test_scenario("algorithmic_filter_bubble")
    settings = scenario.settings()
    
    # Sweep the exaggeration factor to prove the "Filter Bubble" intensity scales
    exaggeration_sweep = [1.0, 1.5, 2.0, 3.0, 4.0]
    mean_engagements = []

    def get_sim_runner(exaggeration):
        def runner():
            # Build society with specific amplification intensity
            society = prepare_scenario_society(
                "algorithmic_filter_bubble",
                tmp_path / f"algo_{exaggeration}_{np.random.randint(1e6)}",
                enable_evolution=False,
                use_algorithmic_amplification=True,
                algo_exaggeration_factor=exaggeration,
                num_agents=200, # Scaled for faster iteration
            )
            world = build_world(settings["world"])
            
            result = run_debug_simulation(
                society.config,
                world,
                society=society,
                urgency=settings["urgency"],
            )
            return result.engagement_scores.mean().item()
        return runner

    # 1. Gradient Sweep
    for ex in exaggeration_sweep:
        results = run_monte_carlo(get_sim_runner(ex), n_seeds=n_seeds)
        mean_engagements.append(np.mean(results))

    # Assertion: As the algorithm becomes more aggressive, engagement must increase
    assert_monotonic_relationship(exaggeration_sweep, mean_engagements, "positive")

    # 2. Statistical Significance: Neutral (1.0) vs Highly Amplified (4.0)
    neutral_results = run_monte_carlo(get_sim_runner(1.0), n_seeds=n_seeds)
    amplified_results = run_monte_carlo(get_sim_runner(4.0), n_seeds=n_seeds)
    
    # Validation: High amplification must create a statistically distinct outcome 
    # compared to no amplification.
    assert_statistically_greater(amplified_results, neutral_results)
