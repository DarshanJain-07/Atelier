import torch

from main import (
    build_debug_society,
    clone_sim_config,
    prepare_society_for_debug,
    run_debug_simulation,
)
from research_paper_tests.config_schema import build_world, get_test_scenario


def test_algorithmic_filter_bubble_mutates_feed_and_boosts_engagement(tmp_path):
    scenario = get_test_scenario("algorithmic_filter_bubble")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "algo"), evolve=False
    )

    boring_world = build_world(settings["world"])

    baseline_config = clone_sim_config(config, use_algorithmic_amplification=False)
    baseline_society = build_debug_society(
        baseline_config,
        society.exposures,
        society.personalities,
        society.affinities,
        society.metadata["Influence"].to_numpy(),
        society.adjacency_matrix,
        society.memory.clone(),
        society.metadata.copy(),
    )

    baseline = run_debug_simulation(
        baseline_config,
        boring_world,
        society=baseline_society,
        urgency=settings["urgency"],
    )
    amplified = run_debug_simulation(
        config,
        boring_world,
        society=society,
        urgency=settings["urgency"],
    )

    assert not torch.allclose(amplified.final_world_tensor, boring_world)
    assert (
        amplified.engagement_scores.mean().item()
        > baseline.engagement_scores.mean().item()
    )
