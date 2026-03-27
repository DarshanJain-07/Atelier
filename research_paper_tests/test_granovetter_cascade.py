import torch

from main import aggregate_social_state, clone_sim_config, create_sim_config, prepare_society_for_debug


def test_granovetter_thresholds_increase_collective_action(tmp_path):
    config = create_sim_config(
        num_agents=500,
        use_network_topology=True,
        use_granovetter_thresholds=True,
        granovetter_threshold_mean=0.2,
        dominant_emotion_threshold=0.1,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "gran"), evolve=False
    )

    emotions = torch.zeros(config.num_agents, 8)
    instigators = int(config.num_agents * 0.05)
    sympathizers = int(config.num_agents * 0.4)
    emotions[:instigators, 6] = 0.8
    emotions[instigators : instigators + sympathizers, 6] = 0.2

    base_config = clone_sim_config(config, use_granovetter_thresholds=False)
    baseline = aggregate_social_state(
        base_config,
        emotions,
        society.metadata["Influence"].to_numpy(),
        engagement_scores=torch.ones(config.num_agents),
        adjacency_matrix=society.adjacency_matrix,
        personalities=society.personalities,
    )
    cascade = aggregate_social_state(
        config,
        emotions,
        society.metadata["Influence"].to_numpy(),
        engagement_scores=torch.ones(config.num_agents),
        adjacency_matrix=society.adjacency_matrix,
        personalities=society.personalities,
    )

    assert cascade["acting_ratio"] > baseline["acting_ratio"]
