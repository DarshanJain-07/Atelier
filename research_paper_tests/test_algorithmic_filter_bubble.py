import torch

from main import DIMENSION_INDICES, build_debug_society, clone_sim_config, create_sim_config, prepare_society_for_debug, run_debug_simulation


def test_algorithmic_filter_bubble_mutates_feed_and_boosts_engagement(tmp_path):
    config = create_sim_config(
        num_agents=500,
        use_algorithmic_amplification=True,
        algo_sample_size=0.1,
        algo_exaggeration_factor=2.0,
        use_network_topology=False,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "algo"), evolve=False
    )

    boring_world = torch.zeros(1, 12)
    boring_world[0, DIMENSION_INDICES["Innovation"]] = 0.2
    boring_world[0, DIMENSION_INDICES["Freedom"]] = -0.1

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
        urgency=0.5,
    )
    amplified = run_debug_simulation(config, boring_world, society=society, urgency=0.5)

    assert not torch.allclose(amplified.final_world_tensor, boring_world)
    assert (
        amplified.engagement_scores.mean().item()
        > baseline.engagement_scores.mean().item()
    )
