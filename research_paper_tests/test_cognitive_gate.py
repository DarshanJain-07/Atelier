import torch

from main import build_debug_society, run_debug_simulation
from research_paper_tests.config_schema import (
    WORLD_DIMENSION_COUNT,
    build_world,
    get_test_scenario,
    set_dimensions,
    set_traits,
    zero_personalities,
)


def test_cognitive_gate_blocks_misaligned_low_openness_agents():
    scenario = get_test_scenario("cognitive_gate")
    config = scenario.sim_config()
    settings = scenario.settings()

    half = config.num_agents // 2
    exposures = torch.zeros(config.num_agents, WORLD_DIMENSION_COUNT)
    personalities = zero_personalities(config.num_agents, fill=settings["trait_fill"])
    set_traits(
        personalities,
        {"Openness": settings["low_openness"]},
        rows=slice(None, half),
    )
    set_traits(
        personalities,
        {"Openness": settings["high_openness"]},
        rows=slice(half, None),
    )

    set_dimensions(exposures, settings["aligned_worldview"], rows=slice(half, None))
    set_dimensions(
        exposures,
        {name: -value for name, value in settings["aligned_worldview"].items()},
        rows=slice(None, half),
    )

    society = build_debug_society(config, exposures, personalities)

    world = build_world(settings["world"])

    result = run_debug_simulation(
        config,
        world,
        society=society,
        urgency=settings["urgency"],
    )
    engagement = result.engagement_scores.numpy()

    assert engagement[:half].mean() < engagement[half:].mean()
    assert float((engagement[:half] == 0.0).mean()) > float((engagement[half:] == 0.0).mean())
