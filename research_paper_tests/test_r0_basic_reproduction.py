import numpy as np

from main import prepare_society_for_debug, run_debug_simulation
from research_paper_tests.config_schema import get_test_scenario


def test_r0_estimate_finds_nonzero_secondary_engagement(tmp_path):
    scenario = get_test_scenario("r0_basic_reproduction")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "r0"), evolve=False
    )

    rng = np.random.default_rng(settings["rng_seed"])
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

    assert np.mean(cascade_sizes) > 0.0
    assert np.std(cascade_sizes) > 0.0
