import os
import inspect

import numpy as np
import scipy.stats as stats
import torch


def get_monte_carlo_seeds():
    """Returns the number of seeds to use for Monte Carlo simulations."""
    return int(os.environ.get("PYTEST_MONTE_CARLO_SEEDS", 5))


def _accepts_seed_argument(run_fn) -> bool:
    try:
        signature = inspect.signature(run_fn)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            return True
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
    return False


def run_monte_carlo(run_fn, n_seeds=None, *, base_seed: int = 1000):
    """
    Runs a simulation function multiple times with different seeds.

    Args:
        run_fn: A callable that executes the simulation and returns a result.
            If it accepts one argument, the current seed is passed explicitly.
        n_seeds: Number of seeds. Defaults to PYTEST_MONTE_CARLO_SEEDS env var or 5.
        base_seed: First seed in the deterministic Monte Carlo sequence.

    Returns:
        A list of results from each run.
    """
    if n_seeds is None:
        n_seeds = get_monte_carlo_seeds()

    accepts_seed = _accepts_seed_argument(run_fn)
    results = []
    for i in range(n_seeds):
        seed = base_seed + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if accepts_seed:
            results.append(run_fn(seed))
        else:
            results.append(run_fn())
    return results


def _as_finite_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        array = array.reshape(-1)
    if array.size == 0:
        raise AssertionError(f"{name} distribution is empty")
    if not np.isfinite(array).all():
        raise AssertionError(f"{name} distribution contains non-finite values: {array}")
    return array


def _is_degenerate(array: np.ndarray, tolerance: float) -> bool:
    return bool(np.allclose(array, array[0], rtol=0.0, atol=tolerance))


def assert_statistically_greater(
    treatment_dist,
    control_dist,
    alpha=0.05,
    *,
    min_effect: float = 0.0,
    paired: bool = True,
    tolerance: float = 1e-12,
):
    """Asserts that treatment results are statistically greater than control."""
    treatment = _as_finite_array(treatment_dist, "treatment")
    control = _as_finite_array(control_dist, "control")

    if len(treatment) <= 1 or len(control) <= 1:
        # Fallback for N=1: simple comparison
        t_mean = np.mean(treatment)
        c_mean = np.mean(control)
        assert t_mean > c_mean + min_effect, (
            f"Treatment mean ({t_mean:.4f}) not greater than control mean "
            f"({c_mean:.4f}) by min_effect={min_effect:.4f} [Fallback N=1]"
        )
        return

    if paired and treatment.shape == control.shape:
        deltas = treatment - control
        mean_delta = float(np.mean(deltas))
        if _is_degenerate(deltas, tolerance):
            assert mean_delta > min_effect, (
                "Paired Monte Carlo deltas are deterministic but not positive enough. "
                f"mean_delta={mean_delta:.6f}, min_effect={min_effect:.6f}"
            )
            return
        t_stat, p_val = stats.ttest_1samp(
            deltas,
            popmean=min_effect,
            alternative="greater",
        )
    else:
        if _is_degenerate(treatment, tolerance) and _is_degenerate(control, tolerance):
            separated = float(np.min(treatment) - np.max(control))
            assert separated > min_effect, (
                "Unpaired distributions are degenerate and not fully separated. "
                f"separation={separated:.6f}, min_effect={min_effect:.6f}"
            )
            return
        t_stat, p_val = stats.ttest_ind(
            treatment,
            control,
            equal_var=False,
            alternative="greater",
        )

    assert not np.isnan(p_val), (
        "Statistical test returned NaN; use more variable seeds or a deterministic "
        "dominance assertion instead of treating this as significant."
    )
    assert p_val < alpha, (
        "Treatment not statistically greater than control. "
        f"p={p_val:.4f}, alpha={alpha}, t={t_stat:.4f}, "
        f"mean_treatment={np.mean(treatment):.4f}, mean_control={np.mean(control):.4f}"
    )


def assert_statistically_different(
    dist_a,
    dist_b,
    alpha=0.05,
    *,
    min_effect: float = 0.0,
    paired: bool = True,
    tolerance: float = 1e-12,
):
    """Asserts that two distributions are statistically different."""
    a = _as_finite_array(dist_a, "dist_a")
    b = _as_finite_array(dist_b, "dist_b")

    if len(a) <= 1 or len(b) <= 1:
        # Fallback for N=1: simple inequality
        assert abs(np.mean(a) - np.mean(b)) > min_effect, (
            "Distributions are not different enough [Fallback N=1]"
        )
        return

    if paired and a.shape == b.shape:
        deltas = a - b
        if _is_degenerate(deltas, tolerance):
            assert abs(float(np.mean(deltas))) > min_effect, (
                "Paired Monte Carlo deltas are deterministic but not different enough. "
                f"mean_delta={np.mean(deltas):.6f}, min_effect={min_effect:.6f}"
            )
            return
        t_stat, p_val = stats.ttest_1samp(deltas, popmean=0.0)
    else:
        if _is_degenerate(a, tolerance) and _is_degenerate(b, tolerance):
            assert abs(float(np.mean(a) - np.mean(b))) > min_effect, (
                "Unpaired distributions are degenerate and not different enough. "
                f"mean_a={np.mean(a):.6f}, mean_b={np.mean(b):.6f}"
            )
            return
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)

    assert not np.isnan(p_val), (
        "Statistical test returned NaN; use more variable seeds or a deterministic "
        "difference assertion instead of treating this as significant."
    )
    assert p_val < alpha, (
        f"Distributions not statistically different. p={p_val:.4f}, "
        f"alpha={alpha}, t={t_stat:.4f}"
    )

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
