import torch

from main import (
    calculate_validation_metrics,
    map_emotions_to_sentiment,
    prepare_society_for_debug,
    run_debug_simulation,
)
from research_paper_tests.config_schema import (
    SENTIMENT_INDICES,
    build_world,
    get_test_scenario,
)


def test_accuracy_metrics_prefer_matching_baseline(tmp_path):
    scenario = get_test_scenario("accuracy_metrics")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "accuracy"), evolve=False
    )

    negative_world = build_world(settings["world"])

    result = run_debug_simulation(
        config,
        negative_world,
        society=society,
        urgency=settings["urgency"],
        is_personal=False,
    )
    emotion_center = result.social_state["objective_center"]
    sentiment = map_emotions_to_sentiment(emotion_center)

    negative_match = calculate_validation_metrics(
        emotion_center,
        settings["matching_baseline"],
    )
    positive_mismatch = calculate_validation_metrics(
        emotion_center,
        settings["mismatched_baseline"],
    )

    assert sentiment[SENTIMENT_INDICES["Negative"]] > sentiment[
        SENTIMENT_INDICES["Positive"]
    ]
    assert (
        negative_match["wasserstein_distance"]
        < positive_mismatch["wasserstein_distance"]
    )
