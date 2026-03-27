from main import create_sim_config, prepare_society_for_debug
from research_paper_tests._metrics import gini


def test_power_law_influence_increases_influence_inequality(tmp_path):
    standard = create_sim_config(
        num_agents=3000,
        use_power_law_influence=False,
        use_network_topology=False,
        enable_evolution=False,
    )
    power = create_sim_config(
        num_agents=3000,
        use_power_law_influence=True,
        use_network_topology=False,
        enable_evolution=False,
    )

    standard_society = prepare_society_for_debug(
        standard, output_dir=str(tmp_path / "standard"), evolve=False
    )
    power_society = prepare_society_for_debug(
        power, output_dir=str(tmp_path / "power"), evolve=False
    )

    assert gini(power_society.metadata["Influence"].to_numpy()) > gini(
        standard_society.metadata["Influence"].to_numpy()
    )
