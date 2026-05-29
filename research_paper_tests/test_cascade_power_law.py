from research_paper_tests._metrics import gini
from research_paper_tests.config_schema import (
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater

def test_power_law_influence_creates_heavier_tail_than_lognormal(tmp_path, n_seeds):
    flat_config = get_test_scenario("cascade_power_law_flat").sim_config()
    power_config = get_test_scenario("cascade_power_law_power").sim_config()

    def runner():
        flat = prepare_scenario_society(
            "cascade_power_law_flat",
            tmp_path,
            enable_evolution=flat_config.enable_evolution,
            output_name="flat",
        )
        power = prepare_scenario_society(
            "cascade_power_law_power",
            tmp_path,
            enable_evolution=power_config.enable_evolution,
            output_name="power",
        )

        flat_influence = flat.metadata["Influence"].to_numpy()
        power_influence = power.metadata["Influence"].to_numpy()

        return {
            "power_gini": gini(power_influence),
            "flat_gini": gini(flat_influence),
            "power_max": power_influence.max(),
            "flat_max": flat_influence.max()
        }

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    power_ginis = [r["power_gini"] for r in results]
    flat_ginis = [r["flat_gini"] for r in results]
    power_maxes = [r["power_max"] for r in results]
    flat_maxes = [r["flat_max"] for r in results]

    assert_statistically_greater(power_ginis, flat_ginis)
    assert_statistically_greater(power_maxes, flat_maxes)
