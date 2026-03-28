import numpy as np
import pandas as pd

from main import PERSONALITY_CORRELATIONS
from research_paper_tests.config_schema import (
    get_test_scenario,
    prepare_scenario_society,
)


def test_generated_personalities_follow_target_correlations(tmp_path):
    scenario = get_test_scenario("personality_correlations")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "personality_correlations",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="correlations",
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
