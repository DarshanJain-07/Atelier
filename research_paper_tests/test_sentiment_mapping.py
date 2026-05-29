import torch

from research_paper_tests.config_schema import (
    SimConfig,
    emotions_to_behavior_aware_sentiment_distribution,
    emotions_to_sentiment_distribution,
)


def test_behavior_aware_sentiment_keeps_low_activity_cases_more_neutral():
    config = SimConfig()
    anger_dominant_emotion = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=torch.float32,
    )

    raw_sentiment = emotions_to_sentiment_distribution(anger_dominant_emotion)
    low_activity_sentiment = emotions_to_behavior_aware_sentiment_distribution(
        anger_dominant_emotion,
        0.10,
        neutral_acting_threshold=config.sentiment_neutrality_acting_threshold,
        activation=config.sentiment_neutrality_activation,
        leaky_slope=config.sentiment_neutrality_leaky_slope,
    )
    high_activity_sentiment = emotions_to_behavior_aware_sentiment_distribution(
        anger_dominant_emotion,
        0.30,
        neutral_acting_threshold=config.sentiment_neutrality_acting_threshold,
        activation=config.sentiment_neutrality_activation,
        leaky_slope=config.sentiment_neutrality_leaky_slope,
    )

    assert low_activity_sentiment[1] > raw_sentiment[1]
    assert low_activity_sentiment[0] < raw_sentiment[0]
    assert low_activity_sentiment[1] > high_activity_sentiment[1]
