from research_paper_tests.config_schema import (
    SESSION_EVOLUTION_MODE_ENV,
    get_test_scenario,
)


def test_session_evolution_override_updates_default_sim_config(monkeypatch):
    monkeypatch.setenv(SESSION_EVOLUTION_MODE_ENV, "with")

    config = get_test_scenario("agent_memory").sim_config()

    assert config.enable_evolution is True


def test_session_evolution_override_updates_default_run_profile(monkeypatch):
    monkeypatch.setenv(SESSION_EVOLUTION_MODE_ENV, "without")

    run = get_test_scenario("agent_memory").run_profile()

    assert run.enable_evolution is False


def test_session_evolution_override_respects_locked_baseline_pairs(monkeypatch):
    monkeypatch.setenv(SESSION_EVOLUTION_MODE_ENV, "with")

    baseline = get_test_scenario("wealth_gini_baseline").sim_config()
    evolved = get_test_scenario("wealth_gini_evolved").sim_config()

    assert baseline.enable_evolution is False
    assert evolved.enable_evolution is True
