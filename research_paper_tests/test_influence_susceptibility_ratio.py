import numpy as np

from main import run_debug_simulation
from research_paper_tests.config_schema import (
    get_test_scenario,
    prepare_scenario_society,
)


def test_structural_influence_improves_realized_reach(tmp_path):
    scenario = get_test_scenario("influence_susceptibility")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "influence_susceptibility",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="reach",
    )

    influences = society.metadata["Influence"].to_numpy()
    mean_influence = influences.mean()
    rng = np.random.default_rng(settings["rng_seed"])
    seed_indices = rng.choice(
        config.num_agents,
        size=settings["sample_size"],
        replace=False,
    )
    realized_reach = []

    for idx in seed_indices:
        thought = society.exposures[idx].unsqueeze(0)
        result = run_debug_simulation(
            config,
            thought,
            society=society,
            urgency=settings["urgency"],
        )
        reach_probability = min(
            1.0,
            settings["reach_probability_base"]
            + (influences[idx] / mean_influence) * settings["reach_probability_gain"],
        )
        sees_post_mask = rng.random(config.num_agents) < reach_probability
        authority_bonus = 1.0 + np.log1p(influences[idx] / mean_influence)
        engaged = result.engagement_scores.detach().cpu().numpy() * authority_bonus
        realized_reach.append(
            float(((engaged > settings["engagement_threshold"]) & sees_post_mask).sum())
        )

    sampled_influence = influences[seed_indices]
    realized_reach = np.asarray(realized_reach, dtype=np.float64)
    top_quartile = realized_reach >= np.percentile(
        realized_reach,
        settings["reach_top_percentile"],
    )

    assert sampled_influence[top_quartile].mean() > sampled_influence[~top_quartile].mean()
