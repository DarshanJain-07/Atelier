import numpy as np
import torch
from research_paper_tests._metrics import mad_metrics

def test_mad_metrics_on_polarized_vs_normal():
    num_agents = 500
    midpoint = num_agents // 2
    rng = np.random.default_rng(42)

    # Normal distribution (no echo chambers)
    normal = rng.normal(0, 1.0, (num_agents, 1))
    normal_tensor = torch.tensor(normal, dtype=torch.float32)
    # Randomly assign communities for the normal group
    normal_partition = {i: rng.integers(0, 2) for i in range(num_agents)}

    normal_metrics = mad_metrics(normal_tensor, normal_partition)

    # Polarized distribution (two distant modes, representing echo chambers)
    polarized = np.concatenate(
        [
            rng.normal(-5.0, 1.0, (midpoint, 1)),
            rng.normal(5.0, 1.0, (num_agents - midpoint, 1)),
        ]
    )
    polarized_tensor = torch.tensor(polarized, dtype=torch.float32)
    # Perfect partition matching the two modes
    polarized_partition = {i: 0 if i < midpoint else 1 for i in range(num_agents)}

    polarized_metrics = mad_metrics(polarized_tensor, polarized_partition)

    print(f"Normal metrics: {normal_metrics}")
    print(f"Polarized metrics: {polarized_metrics}")

    # For a polarized society, between-group distance should be much higher than within-group
    assert polarized_metrics["madgap"] > normal_metrics["madgap"]
    assert polarized_metrics["gdr"] > normal_metrics["gdr"]

    # In a polarized society, GDR should be significantly > 1
    assert polarized_metrics["gdr"] > 2.0
    
    # In a normal society with random partitions, GDR should be around 1.0
    assert 0.8 < normal_metrics["gdr"] < 1.2
