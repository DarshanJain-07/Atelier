import math

import numpy as np
import torch

from main import DIMENSION_INDICES, aggregate_social_state
from research_paper_tests._metrics import bimodality_coefficient
from research_paper_tests.config_schema import (
    EMOTION_DIMENSION_COUNT,
    get_test_scenario,
    prepare_scenario_society,
    set_emotions,
    zero_emotions,
)

THEORETICAL_MAX_DISPERSION = math.sqrt(EMOTION_DIMENSION_COUNT)


def test_bimodality_coefficient_detects_polarized_distribution(tmp_path):
    scenario = get_test_scenario("bimodality_polarization")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "bimodality_polarization",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="bimodality",
    )
    rng = np.random.default_rng(settings["rng_seed"])
    influence = torch.ones(config.num_agents)

    fairness = society.exposures[:, DIMENSION_INDICES["Fairness"]].numpy()
    empirical_bc = bimodality_coefficient(fairness)

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
        ]
    )
    polarized_bc = bimodality_coefficient(polarized)
    normal_bc = bimodality_coefficient(
        rng.normal(settings["normal_mean"], settings["normal_std"], config.num_agents)
    )
    polarized_emotions = zero_emotions(config.num_agents)
    midpoint = config.num_agents // 2
    set_emotions(polarized_emotions, {"Joy": 1.0}, rows=slice(None, midpoint))
    set_emotions(polarized_emotions, {"Anger": 1.0}, rows=slice(midpoint, None))
    dispersion = aggregate_social_state(config, polarized_emotions, influence)["dispersion"]

    assert 0.0 <= empirical_bc <= 1.0
    assert polarized_bc > normal_bc
    assert polarized_bc > settings["min_polarized_bc"]
    assert dispersion <= THEORETICAL_MAX_DISPERSION
