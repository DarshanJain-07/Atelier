import numpy as np
import torch

from main import run_debug_simulation
from research_paper_tests.config_schema import (
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.stats_utils import assert_statistically_greater, run_monte_carlo


def test_r0_estimate_finds_nonzero_secondary_engagement(tmp_path):
    def run_r0():
        scenario = get_test_scenario("r0_basic_reproduction")
        config = scenario.sim_config()
        settings = scenario.settings()
        seed = torch.initial_seed()
        society = prepare_scenario_society(
            "r0_basic_reproduction",
            tmp_path,
            enable_evolution=config.enable_evolution,
            output_name=f"r0_{seed}",
        )

        # Use fixed seed for deterministic choice within a Monte Carlo run
        # but vary it across Monte Carlo runs using the torch seed.
        rng = np.random.default_rng(seed % (2**32))
        seed_indices = rng.choice(
            config.num_agents,
            size=settings["seed_sample_count"],
            replace=False,
        )
        cascade_sizes = []

        for idx in seed_indices:
            thought = society.exposures[idx].unsqueeze(0)
            result = run_debug_simulation(
                config,
                thought,
                society=society,
                urgency=settings["urgency"],
            )
            engaged = (result.engagement_scores > config.cascade_threshold).sum().item() - 1
            cascade_sizes.append(max(0, engaged))

        return np.mean(cascade_sizes)

    r0_means = run_monte_carlo(run_r0)

    # Assert that the mean cascade size is statistically greater than a zero baseline
    assert_statistically_greater(r0_means, [0.0] * len(r0_means))
