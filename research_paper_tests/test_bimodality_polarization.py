import numpy as np

from main import DIMENSION_INDICES, prepare_society_for_debug
from research_paper_tests._metrics import bimodality_coefficient
from research_paper_tests.config_schema import get_test_scenario


def test_bimodality_coefficient_detects_polarized_distribution(tmp_path):
    scenario = get_test_scenario("bimodality_polarization")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "bimodality"), evolve=False
    )
    rng = np.random.default_rng(settings["rng_seed"])

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

    assert 0.0 <= empirical_bc <= 1.0
    assert polarized_bc > normal_bc
    assert polarized_bc > settings["min_polarized_bc"]
