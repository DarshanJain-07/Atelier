from main import create_sim_config, prepare_society_for_debug


def test_personality_distribution_keeps_high_and_low_neuroticism_tails(tmp_path):
    config = create_sim_config(
        num_agents=3000,
        use_network_topology=False,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "tails"), evolve=False
    )

    neuroticism = society.personalities[:, 4].numpy()
    high_share = float((neuroticism > 0.8).mean())
    low_share = float((neuroticism < 0.2).mean())

    assert high_share > 0.01
    assert low_share > 0.01
