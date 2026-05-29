import numpy as np
import scipy.stats as stats
import torch
import os

def get_monte_carlo_seeds():
    """Returns the number of seeds to use for Monte Carlo simulations."""
    return int(os.environ.get("PYTEST_MONTE_CARLO_SEEDS", 5))

def run_monte_carlo(run_fn, n_seeds=None):
    """
    Runs a simulation function multiple times with different seeds.
    
    Args:
        run_fn: A callable that executes the simulation and returns a result (scalar or dict).
        n_seeds: Number of seeds. Defaults to PYTEST_MONTE_CARLO_SEEDS env var or 5.
    
    Returns:
        A list of results from each run.
    """
    if n_seeds is None:
        n_seeds = get_monte_carlo_seeds()
    
    results = []
    for i in range(n_seeds):
        seed = 1000 + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        results.append(run_fn())
    return results

def assert_statistically_greater(treatment_dist, control_dist, alpha=0.05):
    """Asserts that treatment results are statistically greater than control."""
    if len(treatment_dist) <= 1 or len(control_dist) <= 1:
        # Fallback for N=1: simple comparison
        t_mean = np.mean(treatment_dist)
        c_mean = np.mean(control_dist)
        assert t_mean > c_mean, f"Treatment mean ({t_mean:.4f}) not greater than control mean ({c_mean:.4f}) [Fallback N=1]"
        return

    t_stat, p_val = stats.ttest_ind(treatment_dist, control_dist, equal_var=False, alternative='greater')
    # Handle NaN p-values if distributions are identical
    if np.isnan(p_val):
        if np.mean(treatment_dist) > np.mean(control_dist):
            p_val = 0.0
        else:
            p_val = 1.0
            
    assert p_val < alpha, f"Treatment not statistically greater than control. p={p_val:.4f}, alpha={alpha}, t={t_stat:.4f}"

def assert_statistically_different(dist_a, dist_b, alpha=0.05):
    """Asserts that two distributions are statistically different."""
    if len(dist_a) <= 1 or len(dist_b) <= 1:
        # Fallback for N=1: simple inequality
        assert np.mean(dist_a) != np.mean(dist_b), "Distributions are identical [Fallback N=1]"
        return

    t_stat, p_val = stats.ttest_ind(dist_a, dist_b, equal_var=False)
    if np.isnan(p_val):
        p_val = 1.0 if np.mean(dist_a) == np.mean(dist_b) else 0.0

    assert p_val < alpha, f"Distributions not statistically different. p={p_val:.4f}, alpha={alpha}, t={t_stat:.4f}"

def assert_monotonic_relationship(x_sweep, y_results, expected_direction="positive", alpha=0.05):
    """
    Asserts a statistically significant monotonic relationship between x and y 
    using Spearman's rank correlation.
    """
    correlation, p_val = stats.spearmanr(x_sweep, y_results)
    
    # Handle small samples or constant inputs
    if np.isnan(correlation):
        # Fallback for small/constant: check if the first and last values follow the direction
        if expected_direction == "positive":
            assert y_results[-1] > y_results[0], "No positive trend detected [Fallback]"
        else:
            assert y_results[-1] < y_results[0], "No negative trend detected [Fallback]"
        return

    if expected_direction == "positive":
        assert correlation > 0, f"Expected positive correlation, got {correlation:.4f}"
    else:
        assert correlation < 0, f"Expected negative correlation, got {correlation:.4f}"
        
    # Only enforce p-value if we have enough points for significance
    if len(x_sweep) > 3:
        assert p_val < alpha, f"Monotonic relationship not statistically significant. p={p_val:.4f}, correlation={correlation:.4f}"
