from main import run_debug_simulation
from research_paper_tests.config_schema import (
    build_world,
    get_test_scenario,
    prepare_scenario_society,
)


def test_personal_events_stay_more_localized_than_general_events(tmp_path):
    scenario = get_test_scenario("personal")
    config = scenario.sim_config()
    settings = scenario.settings()
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

    assert personal.engagement_scores.mean().item() < general.engagement_scores.mean().item()
