import numpy as np
import torch
from main import run_debug_simulation
from research_paper_tests.config_schema import (
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater


def test_structural_influence_improves_realized_reach(tmp_path, n_seeds):
    """
    RESEARCH FINDINGS - REALIZED REACH
    ----------------------------------
    Validates that agents with higher structural influence (eigenvector centrality 
    proxied by metadata) achieve statistically higher realized reach in simulations.
    """
    scenario = get_test_scenario("influence_susceptibility")
    config = scenario.sim_config()
    settings = scenario.settings()

    def runner():
        society = prepare_scenario_society(
            "influence_susceptibility",
            tmp_path / f"reach_{np.random.randint(1e9)}",
            enable_evolution=config.enable_evolution,
            output_name="reach",
        )

        influences = society.metadata["Influence"].to_numpy()
        mean_influence = influences.mean()
        rng = np.random.default_rng() # Monte Carlo should use fresh RNG
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
                float(((engaged > settings["engagement_threshold"]) & sees_post_mask).sum()),
            )

        sampled_influence = influences[seed_indices]
        realized_reach = np.asarray(realized_reach, dtype=np.float64)
        
        # Calculate means for top and bottom quartiles of reach in this run
        top_quartile_mask = realized_reach >= np.percentile(
            realized_reach,
            settings["reach_top_percentile"],
        )
        
        return (
            sampled_influence[top_quartile_mask].mean(),
            sampled_influence[~top_quartile_mask].mean()
        )

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    top_means = [r[0] for r in results]
    bottom_means = [r[1] for r in results]

    # Statistical Validation: High reach group should have higher mean influence
    assert_statistically_greater(top_means, bottom_means)
    
    print(f"Mean Influence (Top Reach): {np.mean(top_means):.4f}")
    print(f"Mean Influence (Bottom Reach): {np.mean(bottom_means):.4f}")
