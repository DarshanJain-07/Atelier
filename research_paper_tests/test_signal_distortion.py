import numpy as np
import torch

from main import distort_world_signal
from research_paper_tests.config_schema import (
    DIMENSION_INDICES,
    PERSONALITY_INDICES,
    build_world,
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater


def test_signal_distortion_scales_with_neuroticism(tmp_path, n_seeds):
    scenario = get_test_scenario("signal_distortion")
    config = scenario.sim_config()
    settings = scenario.settings()

    def runner():
        # Use a unique output name per seed to avoid file collisions
        import torch
        seed = torch.initial_seed()
        society = prepare_scenario_society(
            "signal_distortion",
            tmp_path,
            enable_evolution=config.enable_evolution,
            output_name=f"distortion_{seed}",
        )

        world = build_world(settings["world"])
        perceived = distort_world_signal(config, world, society.personalities)

        safety_index = DIMENSION_INDICES["Physical_Safety"]
        neuroticism = society.personalities[:, PERSONALITY_INDICES["Neuroticism"]].numpy()
        distortion = np.abs(perceived[:, safety_index].numpy() - world[0, safety_index].item())
        
        correlation = float(np.corrcoef(neuroticism, distortion)[0, 1])
        
        low_cutoff = float(np.quantile(neuroticism, 0.25))
        high_cutoff = float(np.quantile(neuroticism, 0.75))
        
        low_neuroticism_distortion = distortion[neuroticism <= low_cutoff]
        high_neuroticism_distortion = distortion[neuroticism >= high_cutoff]
        
        return {
            "correlation": correlation,
            "low_mean": low_neuroticism_distortion.mean().item(),
            "high_mean": high_neuroticism_distortion.mean().item()
        }

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    correlations = [r["correlation"] for r in results]
    low_means = [r["low_mean"] for r in results]
    high_means = [r["high_mean"] for r in results]

    # Assert correlation is statistically greater than 0
    assert_statistically_greater(correlations, [0.0] * len(correlations))
    
    # Assert high neuroticism group has statistically greater distortion than low neuroticism group
    assert_statistically_greater(high_means, low_means)
