import numpy as np

from main import DIMENSION_INDICES, create_sim_config, prepare_society_for_debug
from research_paper_tests._metrics import bimodality_coefficient


def test_bimodality_coefficient_detects_polarized_distribution(tmp_path):
    config = create_sim_config(
        num_agents=1500,
        use_network_topology=False,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "bimodality"), evolve=False
    )

    fairness = society.exposures[:, DIMENSION_INDICES["Fairness"]].numpy()
    empirical_bc = bimodality_coefficient(fairness)

    polarized = np.concatenate(
        [np.random.normal(-0.8, 0.1, 750), np.random.normal(0.8, 0.1, 750)]
    )
    polarized_bc = bimodality_coefficient(polarized)
    normal_bc = bimodality_coefficient(np.random.normal(0.0, 1.0, 1500))

    assert 0.0 <= empirical_bc <= 1.0
    assert polarized_bc > normal_bc
    assert polarized_bc > 0.555
