import torch

from main import build_debug_society, run_debug_simulation
from schema import DIMENSION_INDICES
from research_paper_tests.config_schema import (
    PERSONALITY_INDICES,
    WORLD_DIMENSION_COUNT,
    build_world,
    get_test_scenario,
    set_dimensions,
    set_traits,
    zero_personalities,
)


def _build_openness_gradient_society(config, settings):
    exposures = torch.zeros(config.num_agents, WORLD_DIMENSION_COUNT)
    personalities = zero_personalities(config.num_agents, fill=settings["trait_fill"])
    personalities[:, PERSONALITY_INDICES["Openness"]] = torch.linspace(
        settings["openness_start"],
        settings["openness_end"],
        config.num_agents,
    )
    worldview_scale = torch.linspace(
        settings["worldview_min_scale"],
        settings["worldview_max_scale"],
        config.num_agents,
    )

    for dimension_name, dimension_value in settings["aligned_worldview"].items():
        exposures[:, DIMENSION_INDICES[dimension_name]] = (
            -dimension_value * worldview_scale
        )

    return build_debug_society(config, exposures, personalities)


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
    assert engagement[:half].mean() < config.engagement_threshold
    assert engagement[half:].mean() > config.engagement_threshold


def test_cognitive_gate_retains_engagement_for_high_openness_gradient():
    scenario = get_test_scenario("figure_cognitive_gate")
    config = scenario.sim_config()
    settings = scenario.settings()

    society = _build_openness_gradient_society(config, settings)
    world = build_world(settings["world"])

    result = run_debug_simulation(
        config,
        world,
        society=society,
        urgency=settings["urgency"],
    )
    engagement = result.engagement_scores
    openness = society.personalities[:, PERSONALITY_INDICES["Openness"]]

    low_band = engagement[: config.num_agents // 4].mean().item()
    high_band = engagement[-config.num_agents // 4 :].mean().item()

    probe_01 = torch.argmin(torch.abs(openness - 0.1)).item()
    probe_03 = torch.argmin(torch.abs(openness - 0.3)).item()
    probe_04 = torch.argmin(torch.abs(openness - 0.4)).item()

    assert engagement.max().item() > 0.05
    assert high_band > low_band + 0.05
    assert engagement[probe_01].item() <= 0.12
    assert engagement[probe_03].item() >= 0.14
    assert engagement[probe_04].item() >= 0.35
