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
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater


def test_semantic_alignment_rewards_matching_sentiment_baselines(tmp_path, n_seeds):
    scenario = get_test_scenario("semantic_alignment")
    config = scenario.sim_config()
    settings = scenario.settings()

    def runner():
        # Use a unique output name per seed to avoid file collisions
        import torch
        seed = torch.initial_seed()
        society = prepare_scenario_society(
            "semantic_alignment",
            tmp_path,
            enable_evolution=config.enable_evolution,
            output_name=f"semantic_{seed}",
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

        positive_sentiment = map_emotions_to_sentiment(
            positive_center,
            positive.social_state["acting_ratio"],
            config=config,
        )
        negative_sentiment = map_emotions_to_sentiment(
            negative_center,
            negative.social_state["acting_ratio"],
            config=config,
        )

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

        return {
            "neg_sent_in_neg_world": negative_sentiment[SENTIMENT_INDICES["Negative"]].item(),
            "neg_sent_in_pos_world": positive_sentiment[SENTIMENT_INDICES["Negative"]].item(),
            "pos_sent_in_pos_world": positive_sentiment[SENTIMENT_INDICES["Positive"]].item(),
            "pos_sent_in_neg_world": negative_sentiment[SENTIMENT_INDICES["Positive"]].item(),
            "pos_match_dist": positive_match["wasserstein_distance"],
            "pos_against_neg_dist": positive_against_negative["wasserstein_distance"],
            "neg_match_dist": negative_match["wasserstein_distance"],
            "neg_against_pos_dist": negative_against_positive["wasserstein_distance"],
        }

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    
    # Assert negative sentiment is higher in negative world than positive world
    assert_statistically_greater(
        [r["neg_sent_in_neg_world"] for r in results],
        [r["neg_sent_in_pos_world"] for r in results]
    )
    
    # Assert positive sentiment is higher in positive world than negative world
    assert_statistically_greater(
        [r["pos_sent_in_pos_world"] for r in results],
        [r["pos_sent_in_neg_world"] for r in results]
    )
    
    # Assert positive match is better (smaller distance) than positive-against-negative
    # So pos_against_neg_dist > pos_match_dist
    assert_statistically_greater(
        [r["pos_against_neg_dist"] for r in results],
        [r["pos_match_dist"] for r in results]
    )
    
    # Assert negative match is better than positive-against-negative
    assert_statistically_greater(
        [r["pos_against_neg_dist"] for r in results],
        [r["neg_match_dist"] for r in results]
    )
    
    # Assert negative match is better than negative-against-positive
    assert_statistically_greater(
        [r["neg_against_pos_dist"] for r in results],
        [r["neg_match_dist"] for r in results]
    )
