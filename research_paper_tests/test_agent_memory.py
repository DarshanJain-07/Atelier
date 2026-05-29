import torch

from main import run_debug_simulation
from research_paper_tests.config_schema import (
    build_world,
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater

def test_agent_memory_accumulates_and_stacks_new_threats(tmp_path, n_seeds):
    scenario = get_test_scenario("agent_memory")
    config = scenario.sim_config()
    settings = scenario.settings()

    def runner():
        society = prepare_scenario_society(
            "agent_memory",
            tmp_path,
            enable_evolution=config.enable_evolution,
            output_name="memory",
        )

        repeated_wealth_threat = build_world(settings["repeat_threat"])

        for _ in range(settings["repeat_count"]):
            run_debug_simulation(
                config,
                repeated_wealth_threat,
                society=society,
                urgency=settings["urgency"],
            )

        new_threat = build_world(settings["new_threat"])

        stacked = run_debug_simulation(
            config,
            new_threat,
            society=society,
            urgency=settings["urgency"],
        )
        fresh_society = prepare_scenario_society(
            "agent_memory",
            tmp_path,
            enable_evolution=config.enable_evolution,
            output_name="fresh",
        )
        fresh = run_debug_simulation(
            config,
            new_threat,
            society=fresh_society,
            urgency=settings["urgency"],
        )
        
        return {
            "stacked_engagement": stacked.engagement_scores.mean().item(),
            "fresh_engagement": fresh.engagement_scores.mean().item()
        }

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    stacked_scores = [r["stacked_engagement"] for r in results]
    fresh_scores = [r["fresh_engagement"] for r in results]
    
    assert_statistically_greater(stacked_scores, fresh_scores)
