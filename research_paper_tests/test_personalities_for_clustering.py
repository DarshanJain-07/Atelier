from research_paper_tests.config_schema import (
    PERSONALITY_INDICES,
    get_test_scenario,
    prepare_scenario_society,
)


def test_personality_distribution_keeps_high_and_low_neuroticism_tails(tmp_path):
    scenario = get_test_scenario("personalities_for_clustering")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "personalities_for_clustering",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="tails",
    )

    neuroticism = society.personalities[:, PERSONALITY_INDICES["Neuroticism"]].numpy()
    high_share = float((neuroticism > settings["high_threshold"]).mean())
    low_share = float((neuroticism < settings["low_threshold"]).mean())

    assert high_share > settings["min_tail_share"]
    assert low_share > settings["min_tail_share"]
