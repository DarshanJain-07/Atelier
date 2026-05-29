import os
import numpy as np
import pytest

from research_paper_tests.config_schema import (
    SOCIETY_EVOLUTION_CASES,
    evolution_variants,
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.stats_utils import run_monte_carlo

EVOLUTION_MATRIX_MODE = os.getenv("RESEARCH_TEST_EVOLUTION_MATRIX", "").strip().lower()

if not EVOLUTION_MATRIX_MODE:
    pytest.skip(
        "Set RESEARCH_TEST_EVOLUTION_MATRIX=with|without|both to run the society "
        "evolution matrix smoke tests.",
        allow_module_level=True,
    )


SELECTED_EVOLUTION_VARIANTS = evolution_variants(EVOLUTION_MATRIX_MODE)


@pytest.mark.parametrize(
    "enable_evolution",
    SELECTED_EVOLUTION_VARIANTS,
    ids=lambda variant: "with_evolution" if variant else "without_evolution",
)
@pytest.mark.parametrize("scenario_name", SOCIETY_EVOLUTION_CASES)
def test_generated_society_cases_support_requested_evolution_modes(
    tmp_path,
    scenario_name,
    enable_evolution,
    n_seeds,
):
    scenario = get_test_scenario(scenario_name)
    expected_config = scenario.sim_config(
        enable_evolution=enable_evolution,
        smoke=True,
    )

    def runner():
        society = prepare_scenario_society(
            scenario_name,
            tmp_path / f"evo_{np.random.randint(1e9)}",
            enable_evolution=enable_evolution,
            smoke=True,
        )

        assert society.config.enable_evolution is enable_evolution
        assert society.exposures.shape[0] == expected_config.num_agents
        assert society.personalities.shape[0] == expected_config.num_agents
        assert society.affinities.shape == society.exposures.shape
        assert society.memory.shape == society.exposures.shape

        if expected_config.use_network_topology:
            assert society.adjacency_matrix is not None
        else:
            assert society.adjacency_matrix is None
        return True

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    assert all(results)
