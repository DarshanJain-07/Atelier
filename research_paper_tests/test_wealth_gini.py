from main import DIMENSION_INDICES, prepare_society_for_debug
from research_paper_tests._metrics import gini
from research_paper_tests.config_schema import get_test_scenario


def test_evolution_increases_wealth_inequality(tmp_path):
    baseline_scenario = get_test_scenario("wealth_gini_baseline")
    baseline = baseline_scenario.sim_config()
    evolved = get_test_scenario("wealth_gini_evolved").sim_config()
    settings = baseline_scenario.settings()

    baseline_society = prepare_society_for_debug(
        baseline, output_dir=str(tmp_path / "baseline"), evolve=False
    )
    evolved_society = prepare_society_for_debug(
        evolved, output_dir=str(tmp_path / "evolved"), evolve=True
    )

    baseline_gini = gini(
        baseline_society.exposures[:, DIMENSION_INDICES["Wealth"]].numpy()
    )
    evolved_gini = gini(
        evolved_society.exposures[:, DIMENSION_INDICES["Wealth"]].numpy()
    )

    assert abs(evolved_gini - baseline_gini) > settings["min_absolute_delta"]
