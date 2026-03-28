from main import prepare_society_for_debug
from research_paper_tests._metrics import gini
from research_paper_tests.config_schema import get_test_scenario


def test_power_law_influence_increases_influence_inequality(tmp_path):
    standard = get_test_scenario("ideological_influence_standard").sim_config()
    power = get_test_scenario("ideological_influence_power").sim_config()

    standard_society = prepare_society_for_debug(
        standard, output_dir=str(tmp_path / "standard"), evolve=False
    )
    power_society = prepare_society_for_debug(
        power, output_dir=str(tmp_path / "power"), evolve=False
    )

    assert gini(power_society.metadata["Influence"].to_numpy()) > gini(
        standard_society.metadata["Influence"].to_numpy()
    )
