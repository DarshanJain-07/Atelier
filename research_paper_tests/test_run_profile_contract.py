from dataclasses import fields

from main import RunProfile, create_sim_config, run_profile_to_sim_config_kwargs
from research_paper_tests.config_schema import live_run_profile_defaults
from schema import RUN_PROFILE_INTERNAL_ONLY_FIELDS, SimConfig


def test_run_profile_defaults_follow_sim_config_defaults():
    run = RunProfile()
    config_kwargs = run_profile_to_sim_config_kwargs(run)
    sim_defaults = SimConfig()

    assert config_kwargs
    for field_name, field_value in config_kwargs.items():
        assert field_value == getattr(sim_defaults, field_name)


def test_run_profile_covers_all_public_sim_config_fields():
    expected_fields = {
        config_field.name
        for config_field in fields(SimConfig)
        if config_field.init and config_field.name not in RUN_PROFILE_INTERNAL_ONLY_FIELDS
    }

    config_kwargs = run_profile_to_sim_config_kwargs(RunProfile())

    assert set(config_kwargs) == expected_fields


def test_run_profile_accepts_sim_config_aliases():
    run = RunProfile(
        num_agents=50,
        mutation_temperature=0.4,
        use_signal_distortion=False,
        use_time_pressure=False,
        use_power_law_influence=True,
    )

    assert run.agent_count == 50
    assert run.temperature == 0.4
    assert run.use_distortion is False
    assert run.use_pressure is False
    assert run.use_power_law is True

    config = create_sim_config(**run_profile_to_sim_config_kwargs(run))
    assert config.num_agents == 50
    assert config.mutation_temperature == 0.4
    assert config.use_signal_distortion is False
    assert config.use_time_pressure is False
    assert config.use_power_law_influence is True


def test_live_run_profile_defaults_match_run_profile_model():
    assert live_run_profile_defaults() == RunProfile().model_dump()
