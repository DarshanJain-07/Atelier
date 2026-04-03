from main import RunProfile, create_sim_config, run_profile_to_sim_config_kwargs
from schema import SimConfig


def test_run_profile_defaults_follow_sim_config_defaults():
    run = RunProfile()
    config_kwargs = run_profile_to_sim_config_kwargs(run)
    sim_defaults = SimConfig()

    assert config_kwargs
    for field_name, field_value in config_kwargs.items():
        assert field_value == getattr(sim_defaults, field_name)


def test_run_profile_accepts_sim_config_aliases():
    run = RunProfile(
        num_agents=50,
        mutation_temperature=0.4,
        use_signal_distortion=False,
        use_time_pressure=False,
        use_maslow_gating=False,
        use_power_law_influence=True,
    )

    assert run.agent_count == 50
    assert run.temperature == 0.4
    assert run.use_distortion is False
    assert run.use_pressure is False
    assert run.use_maslow is False
    assert run.use_power_law is True

    config = create_sim_config(**run_profile_to_sim_config_kwargs(run))
    assert config.num_agents == 50
    assert config.mutation_temperature == 0.4
    assert config.use_signal_distortion is False
    assert config.use_time_pressure is False
    assert config.use_maslow_gating is False
    assert config.use_power_law_influence is True
