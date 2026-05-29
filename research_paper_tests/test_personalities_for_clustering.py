import numpy as np
from research_paper_tests.config_schema import (
    PERSONALITY_INDICES,
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.stats_utils import run_monte_carlo


def test_personality_distribution_keeps_high_and_low_neuroticism_tails(tmp_path, n_seeds):
    scenario = get_test_scenario("personalities_for_clustering")
    config = scenario.sim_config()
    settings = scenario.settings()

    def runner():
        society = prepare_scenario_society(
            "personalities_for_clustering",
            tmp_path,
            enable_evolution=config.enable_evolution,
            output_name="tails",
        )

        neuroticism = society.personalities[:, PERSONALITY_INDICES["Neuroticism"]].numpy()
        
        return {
            "high_q": float(np.quantile(neuroticism, 0.9)),
            "low_q": float(np.quantile(neuroticism, 0.1)),
        }

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    high_qs = [r["high_q"] for r in results]
    low_qs = [r["low_q"] for r in results]

    assert np.mean(high_qs) > settings["high_threshold"]
    assert np.mean(low_qs) < settings["low_threshold"]
