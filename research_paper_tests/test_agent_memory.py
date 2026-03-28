import torch

from main import prepare_society_for_debug, run_debug_simulation
from research_paper_tests.config_schema import build_world, get_test_scenario


def test_agent_memory_accumulates_and_stacks_new_threats(tmp_path):
    scenario = get_test_scenario("agent_memory")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "memory"), evolve=False
    )

    repeated_wealth_threat = build_world(settings["repeat_threat"])

    for _ in range(settings["repeat_count"]):
        run_debug_simulation(
            config,
            repeated_wealth_threat,
            society=society,
            urgency=settings["urgency"],
        )

    assert torch.norm(society.memory).item() > 0.0

    new_threat = build_world(settings["new_threat"])

    stacked = run_debug_simulation(
        config,
        new_threat,
        society=society,
        urgency=settings["urgency"],
    )
    fresh_society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "fresh"), evolve=False
    )
    fresh = run_debug_simulation(
        config,
        new_threat,
        society=fresh_society,
        urgency=settings["urgency"],
    )

    assert stacked.engagement_scores.mean().item() > fresh.engagement_scores.mean().item()
