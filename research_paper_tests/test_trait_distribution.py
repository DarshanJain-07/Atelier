from main import DIMENSION_INDICES, prepare_society_for_debug
from research_paper_tests.config_schema import get_test_scenario


def test_generated_trait_distributions_are_well_formed(tmp_path):
    config = get_test_scenario("trait_distribution").sim_config()
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "traits"), evolve=False
    )

    assert society.personalities.min().item() >= 0.0
    assert society.personalities.max().item() <= 1.0
    assert society.exposures[:, DIMENSION_INDICES["Wealth"]].std().item() > 0.0
    assert society.metadata["Influence"].min() > 0.0
