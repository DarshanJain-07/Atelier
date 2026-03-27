import torch

from main import DIMENSION_INDICES, create_sim_config, prepare_society_for_debug, run_debug_simulation


def test_agent_memory_accumulates_and_stacks_new_threats(tmp_path):
    config = create_sim_config(
        num_agents=300,
        use_agent_memory=True,
        memory_desensitization_gain=5.0,
        memory_trigger_stacking_gain=15.0,
        use_network_topology=False,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "memory"), evolve=False
    )

    repeated_wealth_threat = torch.zeros(1, 12)
    repeated_wealth_threat[0, DIMENSION_INDICES["Wealth"]] = -0.8

    for _ in range(10):
        run_debug_simulation(config, repeated_wealth_threat, society=society, urgency=0.5)

    assert torch.norm(society.memory).item() > 0.0

    new_threat = torch.zeros(1, 12)
    new_threat[0, DIMENSION_INDICES["Physical_Safety"]] = -0.2

    stacked = run_debug_simulation(config, new_threat, society=society, urgency=0.5)
    fresh_society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "fresh"), evolve=False
    )
    fresh = run_debug_simulation(config, new_threat, society=fresh_society, urgency=0.5)

    assert stacked.engagement_scores.mean().item() > fresh.engagement_scores.mean().item()
