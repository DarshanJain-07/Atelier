from main import DIMENSION_INDICES, create_sim_config, prepare_society_for_debug
from research_paper_tests._metrics import gini


def test_evolution_increases_wealth_inequality(tmp_path):
    baseline = create_sim_config(
        num_agents=1500,
        evolution_generations=20,
        use_network_topology=False,
        enable_evolution=False,
    )
    evolved = create_sim_config(
        num_agents=1500,
        evolution_generations=20,
        use_network_topology=False,
        enable_evolution=True,
    )

    baseline_society = prepare_society_for_debug(
        baseline, output_dir=str(tmp_path / "baseline"), evolve=False
    )
    evolved_society = prepare_society_for_debug(
        evolved, output_dir=str(tmp_path / "evolved"), evolve=True
    )

    baseline_gini = gini(
        baseline_society.exposures[:, DIMENSION_INDICES["Wealth"]].numpy()
    )
    evolved_gini = gini(
        evolved_society.exposures[:, DIMENSION_INDICES["Wealth"]].numpy()
    )

    assert abs(evolved_gini - baseline_gini) > 0.02
