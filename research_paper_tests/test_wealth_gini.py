import numpy as np
from research_paper_tests._metrics import gini
from research_paper_tests.config_schema import (
    DIMENSION_INDICES,
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.stats_utils import (
    run_monte_carlo, 
    assert_statistically_greater
)

def test_evolution_increases_wealth_inequality_statistical(tmp_path):
    """
    Statistically validates that evolution increases wealth inequality 
    over multiple Monte Carlo seeds.
    """
    
    def run_baseline():
        # Baseline: No evolution
        society = prepare_scenario_society(
            "wealth_gini_baseline",
            tmp_path,
            enable_evolution=False,
            output_name="baseline",
        )
        return gini(society.exposures[:, DIMENSION_INDICES["Wealth"]].numpy())

    def run_evolved():
        # Evolved: Evolution enabled (generations=20)
        society = prepare_scenario_society(
            "wealth_gini_evolved",
            tmp_path,
            enable_evolution=True,
            output_name="evolved",
        )
        return gini(society.exposures[:, DIMENSION_INDICES["Wealth"]].numpy())

    print("\nRunning Monte Carlo for Wealth Inequality (Gini)...")
    baseline_ginis = run_monte_carlo(run_baseline)
    evolved_ginis = run_monte_carlo(run_evolved)
    
    mean_baseline = np.mean(baseline_ginis)
    mean_evolved = np.mean(evolved_ginis)
    
    print(f"Baseline Gini (mean): {mean_baseline:.4f}")
    print(f"Evolved Gini (mean): {mean_evolved:.4f}")
    
    # 1. Statistical Significance: Evolved > Baseline (p < 0.05)
    assert_statistically_greater(evolved_ginis, baseline_ginis)
    
    # 2. Effect Size: Inequality should be non-trivial in evolved state
    assert mean_evolved > 0.05, f"Evolved Gini {mean_evolved:.4f} below non-trivial threshold"
