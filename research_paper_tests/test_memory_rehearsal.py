import torch

from main import consolidate_agent_memory, create_sim_config


def test_memory_rehearsal_slows_decay():
    config = create_sim_config(
        num_agents=100,
        use_agent_memory=True,
        memory_decay_rate=0.5,
        memory_social_rehearsal_gain=0.8,
        use_network_topology=False,
        enable_evolution=False,
    )

    memory = torch.zeros(config.num_agents, 12)
    context = torch.zeros(config.num_agents, 12)
    context[:, 1] = -1.0

    isolated = consolidate_agent_memory(config, memory, context, social_rehearsal_factor=0.0)
    rehearsed = consolidate_agent_memory(config, memory, context, social_rehearsal_factor=1.0)

    for _ in range(5):
        isolated = consolidate_agent_memory(
            config,
            isolated,
            torch.zeros_like(context),
            social_rehearsal_factor=0.0,
        )
        rehearsed = consolidate_agent_memory(
            config,
            rehearsed,
            torch.zeros_like(context),
            social_rehearsal_factor=1.0,
        )

    assert torch.norm(rehearsed).item() > torch.norm(isolated).item()
