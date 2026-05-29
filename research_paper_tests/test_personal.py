from main import run_debug_simulation
from research_paper_tests.config_schema import (
    build_world,
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater


def test_personal_events_stay_more_localized_than_general_events(tmp_path, n_seeds):
    scenario = get_test_scenario("personal")
    config = scenario.sim_config()
    settings = scenario.settings()

    def runner():
        society = prepare_scenario_society(
            "personal",
            tmp_path,
            enable_evolution=config.enable_evolution,
            output_name="personal",
        )

        world = build_world(settings["world"])

        general = run_debug_simulation(
            config,
            world,
            society=society,
            urgency=settings["urgency"],
            is_personal=False,
        )
        personal = run_debug_simulation(
            config,
            world,
            society=society,
            urgency=settings["urgency"],
            is_personal=True,
        )
        
        return {
            "general_engagement": general.engagement_scores.mean().item(),
            "personal_engagement": personal.engagement_scores.mean().item(),
        }

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    general_scores = [r["general_engagement"] for r in results]
    personal_scores = [r["personal_engagement"] for r in results]

    assert_statistically_greater(general_scores, personal_scores)
