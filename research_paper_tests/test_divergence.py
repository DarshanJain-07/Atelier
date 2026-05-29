import torch
import numpy as np
from main import build_debug_society, project_emotions, run_cognitive_cycle
from research_paper_tests.config_schema import (
    EMOTION_INDICES,
    PERSONALITY_INDICES,
    WORLD_DIMENSION_COUNT,
    build_world,
    get_test_scenario,
    zero_personalities
)
from research_paper_tests.stats_utils import (
    run_monte_carlo,
    assert_monotonic_relationship,
    assert_statistically_greater
)

def test_emotional_divergence_neuroticism_gradient(n_seeds):
    """
    Validation: Higher Neuroticism must lead to higher Fear responses for the same threat signal.
    This proves the engine's Big Five trait mapping follows established psychological theory.
    """
    scenario = get_test_scenario("divergence")
    settings = scenario.settings()
    
    # Sweep Neuroticism to prove the emotional response gradient
    neuroticism_sweep = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    mean_fear_responses = []

    def get_sim_runner(n_val):
        def runner():
            config = scenario.sim_config()
            config.num_agents = 50 # Focused sample
            config.use_signal_distortion = True
            config.distortion_neurotic_gain = 2.5 # Amplify the trait effect
            
            exposures = torch.zeros((config.num_agents, WORLD_DIMENSION_COUNT))
            personalities = zero_personalities(config.num_agents, fill=0.5)
            personalities[:, PERSONALITY_INDICES["Neuroticism"]] = n_val
            
            society = build_debug_society(config, exposures, personalities)
            world = build_world(settings["world"]) # This signal typically contains a threat

            context, _, _ = run_cognitive_cycle(
                config,
                world,
                urgency=settings["urgency"],
                is_personal=False,
                exposures=society.exposures,
                personalities=society.personalities,
                affinities=society.affinities,
            )
            emotions = project_emotions(config, context)
            return emotions[:, EMOTION_INDICES["Fear"]].mean().item()
        return runner

    # 1. Execute Sweep
    for n_val in neuroticism_sweep:
        results = run_monte_carlo(get_sim_runner(n_val), n_seeds=n_seeds)
        mean_fear_responses.append(np.mean(results))

    # Assertion: Increasing Neuroticism must increase the magnitude of the Fear response
    assert_monotonic_relationship(neuroticism_sweep, mean_fear_responses, "positive")

    # 2. Statistical Significance between extreme personality profiles
    low_n_results = run_monte_carlo(get_sim_runner(0.0), n_seeds=n_seeds)
    high_n_results = run_monte_carlo(get_sim_runner(1.0), n_seeds=n_seeds)
    assert_statistically_greater(high_n_results, low_n_results)
