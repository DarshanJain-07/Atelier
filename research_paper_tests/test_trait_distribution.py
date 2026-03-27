from main import DIMENSION_INDICES, create_sim_config, prepare_society_for_debug


def test_generated_trait_distributions_are_well_formed(tmp_path):
    config = create_sim_config(
        num_agents=1200,
        use_network_topology=False,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "traits"), evolve=False
    )

    assert society.personalities.min().item() >= 0.0
    assert society.personalities.max().item() <= 1.0
    assert society.exposures[:, DIMENSION_INDICES["Wealth"]].std().item() > 0.0
    assert society.metadata["Influence"].min() > 0.0
