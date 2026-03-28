import tracemalloc

from main import prepare_society_for_debug, run_debug_simulation
from research_paper_tests.config_schema import build_world, get_test_scenario


def test_full_debug_pipeline_stays_within_reasonable_memory(tmp_path):
    scenario = get_test_scenario("ram_usage")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "ram"), evolve=False
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
