import numpy as np
from sklearn.cluster import KMeans

from main import create_sim_config, prepare_society_for_debug


def test_personality_clusters_show_nontrivial_trait_spread(tmp_path):
    config = create_sim_config(
        num_agents=1200,
        use_network_topology=False,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "clusters"), evolve=False
    )

    personalities = society.personalities.numpy()
    clusters = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(personalities)

    neuroticism_means = [
        float(personalities[clusters == cluster_idx, 4].mean())
        for cluster_idx in range(8)
    ]

    assert np.std(neuroticism_means) > 0.02
