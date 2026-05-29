import numpy as np
from research_paper_tests._metrics import gini
from research_paper_tests.config_schema import (
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater


def test_power_law_influence_increases_influence_inequality(tmp_path, n_seeds):
    """
    RESEARCH FINDINGS - INFLUENCE INEQUALITY
    ----------------------------------------
    Validates that power-law influence distributions consistently 
    create higher Gini coefficients for societal influence than 
    standard distributions.
    """
    def runner():
        # Use randomized subdirectories to avoid state leakage
        std_path = tmp_path / f"std_{np.random.randint(1e9)}"
        pwr_path = tmp_path / f"pwr_{np.random.randint(1e9)}"
        
        standard_society = prepare_scenario_society(
            "ideological_influence_standard",
            std_path,
            enable_evolution=False,
        )
        power_society = prepare_scenario_society(
            "ideological_influence_power",
            pwr_path,
            enable_evolution=False,
        )

        return (
            gini(standard_society.metadata["Influence"].to_numpy()),
            gini(power_society.metadata["Influence"].to_numpy()),
        )

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    standard_ginis = [r[0] for r in results]
    power_ginis = [r[1] for r in results]

    # Statistical Validation
    assert_statistically_greater(power_ginis, standard_ginis)
    
    print(f"Mean Gini (Standard): {np.mean(standard_ginis):.4f}")
    print(f"Mean Gini (Power-Law): {np.mean(power_ginis):.4f}")
