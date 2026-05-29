import torch
import numpy as np
from main import aggregate_social_state, clone_sim_config
from research_paper_tests.config_schema import (
    fraction_count,
    get_test_scenario,
    prepare_scenario_society,
    set_emotions,
    zero_emotions,
)
from research_paper_tests.stats_utils import (
    run_monte_carlo,
    assert_statistically_greater,
    assert_monotonic_relationship
)

def test_granovetter_threshold_cascade_gradient(tmp_path, n_seeds):
    """
    Validation: Collective action must increase monotonically as the percentage 
    of instigator (seed) agents increases. This proves the system follows 
    Granovetter's threshold model of social contagion.
    """
    scenario = get_test_scenario("granovetter_cascade")
    config = scenario.sim_config()
    settings = scenario.settings()
    
    # Sweep the share of initial instigators to observe the cascade tipping point
    instigator_shares = [0.01, 0.03, 0.05, 0.08, 0.12]
    mean_acting_ratios = []

    def get_sim_runner(share):
        def runner():
            society = prepare_scenario_society(
                "granovetter_cascade",
                tmp_path / f"gran_{share}_{np.random.randint(1e6)}",
                enable_evolution=False,
                num_agents=200, # Scaled for faster execution
            )
            emotions = zero_emotions(society.config.num_agents)
            num_instigators = fraction_count(society.config.num_agents, share)
            # Only instigators have the initial 'activating' emotion
            set_emotions(emotions, settings["instigator_emotion"], rows=slice(None, num_instigators))
            
            res = aggregate_social_state(
                society.config,
                emotions,
                society.metadata["Influence"].to_numpy(),
                engagement_scores=torch.ones(society.config.num_agents),
                adjacency_matrix=society.adjacency_matrix,
                personalities=society.personalities,
            )
            return res["acting_ratio"]
        return runner

    # 1. Gradient Sweep
    for share in instigator_shares:
        results = run_monte_carlo(get_sim_runner(share), n_seeds=n_seeds)
        mean_acting_ratios.append(np.mean(results))

    # Assertion: Increasing instigators must increase the final acting ratio
    assert_monotonic_relationship(instigator_shares, mean_acting_ratios, "positive")

    # 2. Statistical Significance: Compare Cascade Model vs Baseline (Linear summation)
    def run_baseline_fn():
        # Baseline where Granovetter threshold logic is disabled
        society = prepare_scenario_society(
            "granovetter_cascade",
            tmp_path / f"base_{np.random.randint(1e6)}",
            enable_evolution=False,
            num_agents=200,
            use_granovetter_thresholds=False
        )
        emotions = zero_emotions(society.config.num_agents)
        num_instigators = fraction_count(society.config.num_agents, 0.05)
        set_emotions(emotions, settings["instigator_emotion"], rows=slice(None, num_instigators))
        res = aggregate_social_state(
            society.config,
            emotions,
            society.metadata["Influence"].to_numpy(),
            engagement_scores=torch.ones(society.config.num_agents),
            adjacency_matrix=society.adjacency_matrix,
            personalities=society.personalities,
        )
        return res["acting_ratio"]

    baseline_results = run_monte_carlo(run_baseline_fn, n_seeds=n_seeds)
    treatment_results = run_monte_carlo(get_sim_runner(0.05), n_seeds=n_seeds)
    
    # Validation: The threshold model should produce higher collective action 
    # than the non-threshold baseline for the same number of instigators.
    assert_statistically_greater(treatment_results, baseline_results)
