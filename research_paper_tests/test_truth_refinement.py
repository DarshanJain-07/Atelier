import torch

from main import build_debug_society, run_cognitive_cycle
from research_paper_tests.config_schema import (
    WORLD_DIMENSION_COUNT,
    build_world,
    get_test_scenario,
)


def test_truth_refinement_prioritizes_long_term_for_skeptical_agents():
    scenario = get_test_scenario("truth_refinement")
    config = scenario.sim_config()
    settings = scenario.settings()
    config.skepticism_gain = settings["skepticism_gain"]
    config.logic_gap_threshold = settings["logic_gap_threshold"]

    world = build_world(settings["world"])
    personalities = torch.tensor(settings["personalities"], dtype=torch.float32)
    society = build_debug_society(
        config,
        torch.zeros(config.num_agents, WORLD_DIMENSION_COUNT),
        personalities,
    )

    _, attention, _ = run_cognitive_cycle(
        config,
        world,
        urgency=0.0,
        is_personal=False,
        exposures=society.exposures,
        personalities=society.personalities,
        affinities=society.affinities,
    )

    skeptical_long_term = attention[1, 11].item()
    skeptical_short_term = attention[1, 10].item()
    populist_long_term = attention[0, 11].item()
    populist_short_term = attention[0, 10].item()

    assert skeptical_long_term > populist_long_term
    assert skeptical_long_term > skeptical_short_term
    assert (skeptical_long_term - skeptical_short_term) > (
        populist_long_term - populist_short_term
    )
