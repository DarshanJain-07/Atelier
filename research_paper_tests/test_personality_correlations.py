import numpy as np
import pandas as pd

from main import PERSONALITY_CORRELATIONS, prepare_society_for_debug
from research_paper_tests.config_schema import get_test_scenario


def test_generated_personalities_follow_target_correlations(tmp_path):
    scenario = get_test_scenario("personality_correlations")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "correlations"), evolve=False
    )

    observed = pd.DataFrame(
        society.personalities.numpy(),
        columns=["O", "C", "E", "A", "N"],
    ).corr()
    target = pd.DataFrame(
        PERSONALITY_CORRELATIONS.numpy(),
        index=["O", "C", "E", "A", "N"],
        columns=["O", "C", "E", "A", "N"],
    )

    rmse = np.sqrt(((observed - target) ** 2).mean().mean())
    assert rmse < settings["max_rmse"]
