import numpy as np
from sklearn.cluster import KMeans

from main import prepare_society_for_debug
from research_paper_tests.config_schema import (
    PERSONALITY_INDICES,
    get_test_scenario,
)


def test_personality_clusters_show_nontrivial_trait_spread(tmp_path):
    scenario = get_test_scenario("clusters")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "clusters"), evolve=False
    )

    personalities = society.personalities.numpy()
    clusters = KMeans(
        n_clusters=settings["cluster_count"],
        random_state=settings["cluster_seed"],
        n_init=settings["cluster_initializations"],
    ).fit_predict(personalities)

    neuroticism_means = [
        float(personalities[clusters == cluster_idx, PERSONALITY_INDICES["Neuroticism"]].mean())
        for cluster_idx in range(settings["cluster_count"])
    ]

    assert np.std(neuroticism_means) > settings["min_neuroticism_spread"]
