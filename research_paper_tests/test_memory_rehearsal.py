import torch

from main import consolidate_agent_memory
from research_paper_tests.config_schema import (
    WORLD_DIMENSION_COUNT,
    get_test_scenario,
    set_dimensions,
)


def test_memory_rehearsal_slows_decay():
    scenario = get_test_scenario("memory_rehearsal")
    config = scenario.sim_config()
    settings = scenario.settings()

    memory = torch.zeros(config.num_agents, WORLD_DIMENSION_COUNT)
    context = torch.zeros(config.num_agents, WORLD_DIMENSION_COUNT)
    set_dimensions(context, settings["context"])

    isolated = consolidate_agent_memory(
        config,
        memory,
        context,
        social_rehearsal_factor=settings["isolated_rehearsal"],
    )
    rehearsed = consolidate_agent_memory(
        config,
        memory,
        context,
        social_rehearsal_factor=settings["shared_rehearsal"],
    )

    for _ in range(settings["decay_steps"]):
        isolated = consolidate_agent_memory(
            config,
            isolated,
            torch.zeros_like(context),
            social_rehearsal_factor=settings["isolated_rehearsal"],
        )
        rehearsed = consolidate_agent_memory(
            config,
            rehearsed,
            torch.zeros_like(context),
            social_rehearsal_factor=settings["shared_rehearsal"],
        )

    assert torch.norm(rehearsed).item() > torch.norm(isolated).item()
