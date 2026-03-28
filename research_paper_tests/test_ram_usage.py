import tracemalloc

from main import run_debug_simulation
from research_paper_tests.config_schema import (
    build_world,
    get_test_scenario,
    prepare_scenario_society,
)


def test_full_debug_pipeline_stays_within_reasonable_memory(tmp_path):
    scenario = get_test_scenario("ram_usage")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "ram_usage",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="ram",
    )

    world = build_world(settings["world"])

    tracemalloc.start()
    run_debug_simulation(
        config,
        world,
        society=society,
        urgency=settings["urgency"],
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak > 0
    assert current <= peak
    assert peak < settings["max_memory_bytes"]
