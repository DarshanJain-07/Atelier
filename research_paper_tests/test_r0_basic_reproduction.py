import numpy as np

from main import create_sim_config, prepare_society_for_debug, run_debug_simulation


def test_r0_estimate_finds_nonzero_secondary_engagement(tmp_path):
    config = create_sim_config(
        num_agents=400,
        use_network_topology=False,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "r0"), evolve=False
    )

    rng = np.random.default_rng(42)
    seed_indices = rng.choice(config.num_agents, size=20, replace=False)
    cascade_sizes = []

    for idx in seed_indices:
        thought = society.exposures[idx].unsqueeze(0)
        result = run_debug_simulation(config, thought, society=society, urgency=0.5)
        engaged = (result.engagement_scores > config.cascade_threshold).sum().item() - 1
        cascade_sizes.append(max(0, engaged))

    assert np.mean(cascade_sizes) > 0.0
    assert np.std(cascade_sizes) > 0.0
