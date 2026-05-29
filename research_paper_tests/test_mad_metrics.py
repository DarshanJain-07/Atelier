import numpy as np
import torch
from research_paper_tests._metrics import mad_metrics
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater

def get_mad_metrics_run(is_polarized=False):
    num_agents = 500
    midpoint = num_agents // 2
    
    if not is_polarized:
        # Normal distribution
        data = np.random.normal(0, 1.0, (num_agents, 1))
        partition = {i: np.random.randint(0, 2) for i in range(num_agents)}
    else:
        # Polarized distribution
        data = np.concatenate(
            [
                np.random.normal(-5.0, 1.0, (midpoint, 1)),
                np.random.normal(5.0, 1.0, (num_agents - midpoint, 1)),
            ],
        )
        partition = {i: 0 if i < midpoint else 1 for i in range(num_agents)}
        
    tensor = torch.tensor(data, dtype=torch.float32)
    metrics = mad_metrics(tensor, partition)
    return metrics

def test_mad_metrics_statistical():
    """
    Goal: Statistically verify that MAD metrics correctly identify polarized societies.
    """
    print("\nRunning Monte Carlo for MAD Metrics...")
    
    normal_results = run_monte_carlo(lambda: get_mad_metrics_run(is_polarized=False))
    polarized_results = run_monte_carlo(lambda: get_mad_metrics_run(is_polarized=True))
    
    normal_madgaps = [m["madgap"] for m in normal_results]
    polarized_madgaps = [m["madgap"] for m in polarized_results]
    
    normal_gdrs = [m["gdr"] for m in normal_results]
    polarized_gdrs = [m["gdr"] for m in polarized_results]
    
    print(f"Normal Mean GDR: {np.mean(normal_gdrs):.3f}")
    print(f"Polarized Mean GDR: {np.mean(polarized_gdrs):.3f}")
    
    # Polarized society should have significantly higher madgap and GDR
    assert_statistically_greater(polarized_madgaps, normal_madgaps)
    assert_statistically_greater(polarized_gdrs, normal_gdrs)
    
    # Absolute baselines
    assert np.mean(polarized_gdrs) > 2.0
    assert 0.8 < np.mean(normal_gdrs) < 1.2

if __name__ == "__main__":
    test_mad_metrics_statistical()
