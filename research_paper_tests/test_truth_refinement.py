import torch
import numpy as np

from main import build_debug_society, run_cognitive_cycle
from research_paper_tests.config_schema import (
    WORLD_DIMENSION_COUNT,
    build_world,
    get_test_scenario,
)
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater


def test_truth_refinement_prioritizes_long_term_for_skeptical_agents():
    """
    Validates that skeptical agents prioritize long-term framing over short-term framing,
    and do so more than populist agents, using Monte Carlo trials and Welch's t-test.
    """
    scenario = get_test_scenario("truth_refinement")
    config = scenario.sim_config()
    settings = scenario.settings()
    config.skepticism_gain = settings["skepticism_gain"]
    config.logic_gap_threshold = settings["logic_gap_threshold"]

    world = build_world(settings["world"])
    personalities_base = torch.tensor(settings["personalities"], dtype=torch.float32)

    def run_truth_trial():
        society = build_debug_society(
            config,
            torch.zeros(config.num_agents, WORLD_DIMENSION_COUNT),
            personalities_base,
        )

        _, attention, _ = run_cognitive_cycle(
            config,
            world,
            urgency=0.0,
            is_personal=False,
            exposures=society.exposures,
            personalities=society.personalities,
            affinities=society.affinities,
        )
        
        return (
            attention[1, 11].item(), # skeptical_long_term
            attention[0, 11].item(), # populist_long_term
            attention[1, 10].item(), # skeptical_short_term
            attention[0, 10].item()  # populist_short_term
        )

    print("Running Truth Refinement Monte Carlo trials...")
    results = run_monte_carlo(run_truth_trial)
    skp_lt = [r[0] for r in results]
    pop_lt = [r[1] for r in results]
    skp_st = [r[2] for r in results]
    pop_st = [r[3] for r in results]

    # 1. Skeptical agents prioritize long-term more than populists do
    assert_statistically_greater(skp_lt, pop_lt)
    
    # 2. Skeptical agents prioritize long-term over their own short-term
    assert_statistically_greater(skp_lt, skp_st)
    
    # 3. The long-short gap is statistically larger for skeptical agents than for populists
    skp_gap = [r[0] - r[2] for r in results]
    pop_gap = [r[1] - r[3] for r in results]
    assert_statistically_greater(skp_gap, pop_gap)
    
    print(f"Results: Skp LT Mean = {np.mean(skp_lt):.4f}, Pop LT Mean = {np.mean(pop_lt):.4f} (p < 0.05)")
