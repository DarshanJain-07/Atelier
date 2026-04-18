from main import DIMENSION_INDICES
from research_paper_tests._metrics import gini
from research_paper_tests.config_schema import (
    get_test_scenario,
    prepare_scenario_society,
)


def test_evolution_increases_wealth_inequality(tmp_path):
    baseline_scenario = get_test_scenario("wealth_gini_baseline")
    baseline = baseline_scenario.sim_config()
    evolved = get_test_scenario("wealth_gini_evolved").sim_config()

    baseline_society = prepare_scenario_society(
        "wealth_gini_baseline",
        tmp_path,
        enable_evolution=baseline.enable_evolution,
        output_name="baseline",
    )
    evolved_society = prepare_scenario_society(
        "wealth_gini_evolved",
        tmp_path,
        enable_evolution=evolved.enable_evolution,
        output_name="evolved",
    )

    baseline_gini = gini(
        baseline_society.exposures[:, DIMENSION_INDICES["Wealth"]].numpy(),
    )
    evolved_gini = gini(
        evolved_society.exposures[:, DIMENSION_INDICES["Wealth"]].numpy(),
    )

    # Downstream cognition consumes normalized wealth exposure, so the regression
    # check is framed around the exposure-space inequality the engine actually uses.
    assert evolved_gini > baseline_gini
