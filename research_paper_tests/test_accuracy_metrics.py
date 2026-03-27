import torch

from main import calculate_validation_metrics, create_sim_config, map_emotions_to_sentiment, prepare_society_for_debug, run_debug_simulation


def test_accuracy_metrics_prefer_matching_baseline(tmp_path):
    config = create_sim_config(
        num_agents=256,
        use_signal_distortion=False,
        use_network_topology=False,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "accuracy"), evolve=False
    )

    negative_world = torch.zeros(1, 12)
    negative_world[0, 1] = -0.8

    result = run_debug_simulation(
        config,
        negative_world,
        society=society,
        urgency=0.5,
        is_personal=False,
    )
    emotion_center = result.social_state["objective_center"]
    sentiment = map_emotions_to_sentiment(emotion_center)

    negative_match = calculate_validation_metrics(emotion_center, [0.9, 0.05, 0.05])
    positive_mismatch = calculate_validation_metrics(emotion_center, [0.05, 0.05, 0.9])

    assert sentiment[0] > sentiment[2]
    assert (
        negative_match["wasserstein_distance"]
        < positive_mismatch["wasserstein_distance"]
    )
