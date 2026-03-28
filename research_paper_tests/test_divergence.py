import torch

from main import build_debug_society, project_emotions, run_cognitive_cycle
from research_paper_tests.config_schema import (
    EMOTION_INDICES,
    WORLD_DIMENSION_COUNT,
    build_world,
    get_test_scenario,
)


def test_emotional_divergence_tracks_neuroticism():
    scenario = get_test_scenario("divergence")
    config = scenario.sim_config()
    settings = scenario.settings()
    personalities = torch.tensor(settings["personalities"], dtype=torch.float32)
    exposures = torch.zeros((config.num_agents, WORLD_DIMENSION_COUNT))
    society = build_debug_society(config, exposures, personalities)

    world = build_world(settings["world"])

    context, _, _ = run_cognitive_cycle(
        config,
        world,
        urgency=settings["urgency"],
        is_personal=False,
        exposures=society.exposures,
        personalities=society.personalities,
        affinities=society.affinities,
    )
    emotions = project_emotions(config, context)

    assert emotions[1, EMOTION_INDICES["Fear"]].item() > emotions[
        0, EMOTION_INDICES["Fear"]
    ].item()
