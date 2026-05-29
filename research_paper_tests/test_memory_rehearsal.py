import torch
import numpy as np
from main import consolidate_agent_memory
from research_paper_tests.config_schema import (
    WORLD_DIMENSION_COUNT,
    get_test_scenario,
    set_dimensions,
)
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater

def run_memory_rehearsal(is_shared=False):
    scenario = get_test_scenario("memory_rehearsal")
    config = scenario.sim_config()
    settings = scenario.settings()

    memory = torch.zeros(config.num_agents, WORLD_DIMENSION_COUNT)
    context = torch.zeros(config.num_agents, WORLD_DIMENSION_COUNT)
    set_dimensions(context, settings["context"])

    rehearsal_factor = settings["shared_rehearsal"] if is_shared else settings["isolated_rehearsal"]

    current_memory = consolidate_agent_memory(
        config,
        memory,
        context,
        social_rehearsal_factor=rehearsal_factor,
    )

    for _ in range(settings["decay_steps"]):
        current_memory = consolidate_agent_memory(
            config,
            current_memory,
            torch.zeros_like(context),
            social_rehearsal_factor=rehearsal_factor,
        )
    
    return torch.norm(current_memory).item()

def test_memory_rehearsal_statistical():
    """
    Goal: Statistically verify that social rehearsal slows down memory decay.
    """
    print("\nRunning Monte Carlo for Memory Rehearsal...")
    
    isolated_norms = run_monte_carlo(lambda: run_memory_rehearsal(is_shared=False))
    shared_norms = run_monte_carlo(lambda: run_memory_rehearsal(is_shared=True))
    
    print(f"Isolated Mean Norm: {np.mean(isolated_norms):.3f}")
    print(f"Shared Mean Norm: {np.mean(shared_norms):.3f}")
    
    # Shared rehearsal should lead to significantly higher remaining memory norm
    assert_statistically_greater(shared_norms, isolated_norms)

if __name__ == "__main__":
    test_memory_rehearsal_statistical()
