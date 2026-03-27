from main import create_sim_config, prepare_society_for_debug
from research_paper_tests._metrics import gini


def test_power_law_influence_creates_heavier_tail_than_lognormal(tmp_path):
    flat_config = create_sim_config(
        num_agents=2000,
        use_power_law_influence=False,
        use_network_topology=False,
        enable_evolution=False,
    )
    power_config = create_sim_config(
        num_agents=2000,
        use_power_law_influence=True,
        use_network_topology=False,
        enable_evolution=False,
    )

    flat = prepare_society_for_debug(
        flat_config, output_dir=str(tmp_path / "flat"), evolve=False
    )
    power = prepare_society_for_debug(
        power_config, output_dir=str(tmp_path / "power"), evolve=False
    )

    flat_influence = flat.metadata["Influence"].to_numpy()
    power_influence = power.metadata["Influence"].to_numpy()

    assert gini(power_influence) > gini(flat_influence)
    assert power_influence.max() > flat_influence.max()
