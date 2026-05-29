import numpy as np
from sklearn.cluster import KMeans

from research_paper_tests.config_schema import (
    PERSONALITY_INDICES,
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.stats_utils import (
    run_monte_carlo,
    assert_statistically_greater,
)


def test_personality_clusters_show_nontrivial_trait_spread(tmp_path, n_seeds):
    scenario = get_test_scenario("clusters")
    config = scenario.sim_config()
    settings = scenario.settings()

    def runner():
        society = prepare_scenario_society(
            "clusters",
            tmp_path / f"clusters_{np.random.randint(1e9)}",
            enable_evolution=config.enable_evolution,
            num_agents=config.num_agents,
        )

        personalities = society.personalities.numpy()
        clusters = KMeans(
            n_clusters=settings["cluster_count"],
            n_init=settings["cluster_initializations"],
        ).fit_predict(personalities)

        neuroticism_means = [
            float(
                personalities[
                    clusters == cluster_idx, PERSONALITY_INDICES["Neuroticism"]
                ].mean()
            )
            for cluster_idx in range(settings["cluster_count"])
        ]
        actual_spread = float(np.std(neuroticism_means))

        # Calculate a single random spread for this society's personality distribution
        rng = np.random.default_rng()
        shuffled_clusters = clusters.copy()
        rng.shuffle(shuffled_clusters)
        neuroticism = personalities[:, PERSONALITY_INDICES["Neuroticism"]]
        shuffled_means = [
            float(neuroticism[shuffled_clusters == cluster_idx].mean())
            for cluster_idx in range(settings["cluster_count"])
        ]
        random_spread = float(np.std(shuffled_means))

        return actual_spread, random_spread

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    actual_dist = [r[0] for r in results]
    random_dist = [r[1] for r in results]

    assert_statistically_greater(actual_dist, random_dist)
