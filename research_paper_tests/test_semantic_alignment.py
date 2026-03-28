from main import (
    calculate_validation_metrics,
    map_emotions_to_sentiment,
    run_debug_simulation,
)
from research_paper_tests.config_schema import (
    SENTIMENT_INDICES,
    build_world,
    get_test_scenario,
    prepare_scenario_society,
)


def test_semantic_alignment_rewards_matching_sentiment_baselines(tmp_path):
    scenario = get_test_scenario("semantic_alignment")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "semantic_alignment",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="semantic",
    )

    positive_world = build_world(settings["positive_world"])
    negative_world = build_world(settings["negative_world"])

    positive = run_debug_simulation(
        config,
        positive_world,
        society=society,
        urgency=settings["urgency"],
    )
    negative = run_debug_simulation(
        config,
        negative_world,
        society=society,
        urgency=settings["urgency"],
    )

    positive_center = positive.social_state["objective_center"]
    negative_center = negative.social_state["objective_center"]

    positive_sentiment = map_emotions_to_sentiment(positive_center)
    negative_sentiment = map_emotions_to_sentiment(negative_center)

    positive_match = calculate_validation_metrics(
        positive_center,
        settings["positive_sentiment_profile"],
    )
    positive_against_negative = calculate_validation_metrics(
        positive_center,
        settings["negative_sentiment_profile"],
    )
    negative_match = calculate_validation_metrics(
        negative_center,
        settings["negative_sentiment_profile"],
    )
    negative_against_positive = calculate_validation_metrics(
        negative_center,
        settings["positive_sentiment_profile"],
    )

    assert negative_sentiment[SENTIMENT_INDICES["Negative"]] > positive_sentiment[
        SENTIMENT_INDICES["Negative"]
    ]
    assert negative_sentiment[SENTIMENT_INDICES["Positive"]] < positive_sentiment[
        SENTIMENT_INDICES["Positive"]
    ]
    assert negative_match["wasserstein_distance"] < positive_against_negative["wasserstein_distance"]
    assert negative_against_positive["wasserstein_distance"] > positive_against_negative["wasserstein_distance"]
