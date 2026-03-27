import numpy as np

from main import create_sim_config, prepare_society_for_debug, run_debug_simulation


def test_structural_influence_improves_realized_reach(tmp_path):
    config = create_sim_config(
        num_agents=800,
        use_power_law_influence=True,
        use_network_topology=False,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "reach"), evolve=False
    )

    influences = society.metadata["Influence"].to_numpy()
    mean_influence = influences.mean()
    rng = np.random.default_rng(42)
    seed_indices = rng.choice(config.num_agents, size=120, replace=False)
    realized_reach = []

    for idx in seed_indices:
        thought = society.exposures[idx].unsqueeze(0)
        result = run_debug_simulation(config, thought, society=society, urgency=0.5)
        reach_probability = min(1.0, 0.10 + (influences[idx] / mean_influence) * 0.10)
        sees_post_mask = rng.random(config.num_agents) < reach_probability
        authority_bonus = 1.0 + np.log1p(influences[idx] / mean_influence)
        engaged = result.engagement_scores.detach().cpu().numpy() * authority_bonus
        realized_reach.append(float(((engaged > 0.18) & sees_post_mask).sum()))

    sampled_influence = influences[seed_indices]
    realized_reach = np.asarray(realized_reach, dtype=np.float64)
    top_quartile = realized_reach >= np.percentile(realized_reach, 75)

    assert sampled_influence[top_quartile].mean() > sampled_influence[~top_quartile].mean()
