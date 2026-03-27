import torch

from main import (
    calculate_validation_metrics,
    create_sim_config,
    map_emotions_to_sentiment,
    prepare_society_for_debug,
    run_debug_simulation,
)


def test_semantic_alignment_rewards_matching_sentiment_baselines(tmp_path):
    config = create_sim_config(
        num_agents=256,
        use_signal_distortion=False,
        use_network_topology=False,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "semantic"), evolve=False
    )

    positive_world = torch.zeros(1, 12)
    positive_world[0, 0] = 0.8
    positive_world[0, 6] = 0.6

    negative_world = torch.zeros(1, 12)
    negative_world[0, 1] = -0.8
    negative_world[0, 4] = -0.6

    positive = run_debug_simulation(config, positive_world, society=society, urgency=0.5)
    negative = run_debug_simulation(config, negative_world, society=society, urgency=0.5)

    positive_center = positive.social_state["objective_center"]
    negative_center = negative.social_state["objective_center"]

    positive_sentiment = map_emotions_to_sentiment(positive_center)
    negative_sentiment = map_emotions_to_sentiment(negative_center)

    positive_match = calculate_validation_metrics(positive_center, [0.05, 0.1, 0.85])
    positive_against_negative = calculate_validation_metrics(
        positive_center, [0.85, 0.1, 0.05]
    )
    negative_match = calculate_validation_metrics(negative_center, [0.85, 0.1, 0.05])
    negative_against_positive = calculate_validation_metrics(
        negative_center, [0.05, 0.1, 0.85]
    )

    assert negative_sentiment[0] > positive_sentiment[0]
    assert negative_sentiment[2] < positive_sentiment[2]
    assert negative_match["wasserstein_distance"] < positive_against_negative["wasserstein_distance"]
    assert negative_against_positive["wasserstein_distance"] > positive_against_negative["wasserstein_distance"]
