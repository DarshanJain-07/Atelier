import torch

from main import build_debug_society, run_debug_simulation
from research_paper_tests.config_schema import (
    EMOTION_INDICES,
    WORLD_DIMENSION_COUNT,
    build_world,
    get_test_scenario,
    set_dimensions,
    set_traits,
    zero_personalities,
)
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater


def test_relative_deprivation_hits_marginalized_agents_harder(n_seeds):
    scenario = get_test_scenario("relative_deprivation")
    config = scenario.sim_config()
    settings = scenario.settings()

    def runner():
        exposures_marginalized = torch.zeros(settings["group_size"], WORLD_DIMENSION_COUNT)
        set_dimensions(exposures_marginalized, settings["marginalized_exposures"])

        personalities_marginalized = zero_personalities(
            settings["group_size"],
            fill=settings["trait_fill"],
        )
        set_traits(personalities_marginalized, settings["marginalized_traits"])

        exposures_elites = torch.zeros(settings["group_size"], WORLD_DIMENSION_COUNT)
        set_dimensions(exposures_elites, settings["elite_exposures"])

        personalities_elites = zero_personalities(
            settings["group_size"],
            fill=settings["trait_fill"],
        )
        set_traits(personalities_elites, settings["elite_traits"])

        society = build_debug_society(
            config,
            torch.cat([exposures_marginalized, exposures_elites], dim=0),
            torch.cat([personalities_marginalized, personalities_elites], dim=0),
        )

        world = build_world(settings["world"])

        result = run_debug_simulation(
            config,
            world,
            society=society,
            urgency=settings["urgency"],
        )
        anger = result.final_emotions[:, EMOTION_INDICES["Anger"]]
        
        return {
            "marginalized_anger": anger[: settings["group_size"]].mean().item(),
            "elite_anger": anger[settings["group_size"] :].mean().item()
        }

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    marginalized_angers = [r["marginalized_anger"] for r in results]
    elite_angers = [r["elite_anger"] for r in results]

    assert_statistically_greater(marginalized_angers, elite_angers)
