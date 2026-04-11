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

    off_diagonal_mask = ~np.eye(len(target), dtype=bool)
    rmse_to_target = float(
        np.sqrt(((observed - target).to_numpy()[off_diagonal_mask] ** 2).mean())
    )
    rmse_to_zero = float(
        np.sqrt((observed.to_numpy()[off_diagonal_mask] ** 2).mean())
    )

    observed_pairs = {
        (left, right): float(observed.loc[left, right])
        for left, right in [("O", "E"), ("C", "A"), ("C", "N"), ("A", "N")]
    }
    top_pairs = sorted(
        (
            ((left, right), abs(float(observed.loc[left, right])))
            for left, right in [
                ("O", "C"),
                ("O", "E"),
                ("O", "A"),
                ("O", "N"),
                ("C", "E"),
                ("C", "A"),
                ("C", "N"),
                ("E", "A"),
                ("E", "N"),
                ("A", "N"),
            ]
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    assert rmse_to_target < rmse_to_zero
    assert set(pair for pair, _ in top_pairs[:4]) == {
        ("O", "E"),
        ("C", "A"),
        ("C", "N"),
        ("A", "N"),
    }
    assert observed_pairs[("O", "E")] > 0.0
    assert observed_pairs[("C", "A")] > 0.0
    assert observed_pairs[("C", "N")] < 0.0
    assert observed_pairs[("A", "N")] < 0.0
