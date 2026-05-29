import math
import numpy as np
import torch

from main import aggregate_social_state
from research_paper_tests._metrics import bimodality_coefficient
from research_paper_tests.config_schema import (
    DIMENSION_INDICES,
    EMOTION_DIMENSION_COUNT,
    get_test_scenario,
    prepare_scenario_society,
    set_emotions,
    zero_emotions,
)
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater

THEORETICAL_MAX_DISPERSION = math.sqrt(EMOTION_DIMENSION_COUNT)

def test_bimodality_coefficient_detects_polarized_distribution(tmp_path, n_seeds):
    scenario = get_test_scenario("bimodality_polarization")
    config = scenario.sim_config()
    settings = scenario.settings()

    def runner():
        society = prepare_scenario_society(
            "bimodality_polarization",
            tmp_path,
            enable_evolution=config.enable_evolution,
            output_name="bimodality",
        )
        # np.random is already seeded by run_monte_carlo
        rng = np.random.default_rng()

        polarized = np.concatenate(
            [
                rng.normal(
                    settings["polarized_mean_negative"],
                    settings["polarized_std"],
                    settings["polarized_count_per_mode"],
                ),
                rng.normal(
                    settings["polarized_mean_positive"],
                    settings["polarized_std"],
                    settings["polarized_count_per_mode"],
                ),
            ],
        )
        polarized_bc = bimodality_coefficient(polarized)
        normal_bc = bimodality_coefficient(
            rng.normal(settings["normal_mean"], settings["normal_std"], config.num_agents),
        )
        
        return {
            "polarized_bc": polarized_bc,
            "normal_bc": normal_bc
        }

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    polarized_bcs = [r["polarized_bc"] for r in results]
    normal_bcs = [r["normal_bc"] for r in results]
    
    assert_statistically_greater(polarized_bcs, normal_bcs)

    # Deterministic sanity checks
    polarized_emotions = zero_emotions(config.num_agents)
    midpoint = config.num_agents // 2
    set_emotions(polarized_emotions, {"Joy": 1.0}, rows=slice(None, midpoint))
    set_emotions(polarized_emotions, {"Anger": 1.0}, rows=slice(midpoint, None))
    influence = torch.ones(config.num_agents)
    dispersion = aggregate_social_state(config, polarized_emotions, influence)["dispersion"]
    assert dispersion <= THEORETICAL_MAX_DISPERSION
