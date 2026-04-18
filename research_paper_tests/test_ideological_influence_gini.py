from research_paper_tests._metrics import gini
from research_paper_tests.config_schema import (
    get_test_scenario,
    prepare_scenario_society,
)


def test_power_law_influence_increases_influence_inequality(tmp_path):
    standard = get_test_scenario("ideological_influence_standard").sim_config()
    power = get_test_scenario("ideological_influence_power").sim_config()

    standard_society = prepare_scenario_society(
        "ideological_influence_standard",
        tmp_path,
        enable_evolution=standard.enable_evolution,
        output_name="standard",
    )
    power_society = prepare_scenario_society(
        "ideological_influence_power",
        tmp_path,
        enable_evolution=power.enable_evolution,
        output_name="power",
    )

    assert gini(power_society.metadata["Influence"].to_numpy()) > gini(
        standard_society.metadata["Influence"].to_numpy(),
    )
