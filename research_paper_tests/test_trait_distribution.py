from main import DIMENSION_INDICES
from research_paper_tests.config_schema import (
    get_test_scenario,
    prepare_scenario_society,
)


def test_generated_trait_distributions_are_well_formed(tmp_path):
    config = get_test_scenario("trait_distribution").sim_config()
    society = prepare_scenario_society(
        "trait_distribution",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="traits",
    )

    assert society.personalities.min().item() >= 0.0
    assert society.personalities.max().item() <= 1.0
    assert society.exposures[:, DIMENSION_INDICES["Wealth"]].std().item() > 0.0
    assert society.metadata["Influence"].min() > 0.0
