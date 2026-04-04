import torch

from main import aggregate_social_state, clone_sim_config
from research_paper_tests.config_schema import (
    fraction_count,
    get_test_scenario,
    prepare_scenario_society,
    set_emotions,
    zero_emotions,
)


def test_granovetter_thresholds_increase_collective_action(tmp_path):
    scenario = get_test_scenario("granovetter_cascade")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "granovetter_cascade",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="gran",
    )

    emotions = zero_emotions(config.num_agents)
    instigators = fraction_count(config.num_agents, settings["instigator_share"])
    sympathizers = fraction_count(config.num_agents, settings["sympathizer_share"])
    set_emotions(emotions, settings["instigator_emotion"], rows=slice(None, instigators))
    set_emotions(
        emotions,
        settings["sympathizer_emotion"],
        rows=slice(instigators, instigators + sympathizers),
    )

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

    assert cascade["acting_ratio"] >= baseline["acting_ratio"]
    if not config.enable_evolution:
        assert cascade["acting_ratio"] > baseline["acting_ratio"]
