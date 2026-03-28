from research_paper_tests._metrics import gini
from research_paper_tests.config_schema import (
    get_test_scenario,
    prepare_scenario_society,
)


def test_power_law_influence_creates_heavier_tail_than_lognormal(tmp_path):
    flat_config = get_test_scenario("cascade_power_law_flat").sim_config()
    power_config = get_test_scenario("cascade_power_law_power").sim_config()

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

    assert gini(power_influence) > gini(flat_influence)
    assert power_influence.max() > flat_influence.max()
