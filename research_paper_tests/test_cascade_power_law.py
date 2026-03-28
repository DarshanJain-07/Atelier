from main import prepare_society_for_debug
from research_paper_tests._metrics import gini
from research_paper_tests.config_schema import get_test_scenario


def test_power_law_influence_creates_heavier_tail_than_lognormal(tmp_path):
    flat_config = get_test_scenario("cascade_power_law_flat").sim_config()
    power_config = get_test_scenario("cascade_power_law_power").sim_config()

    flat = prepare_society_for_debug(
        flat_config, output_dir=str(tmp_path / "flat"), evolve=False
    )
    power = prepare_society_for_debug(
        power_config, output_dir=str(tmp_path / "power"), evolve=False
    )

    flat_influence = flat.metadata["Influence"].to_numpy()
    power_influence = power.metadata["Influence"].to_numpy()

    assert gini(power_influence) > gini(flat_influence)
    assert power_influence.max() > flat_influence.max()
