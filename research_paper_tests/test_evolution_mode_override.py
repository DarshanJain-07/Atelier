import pytest
from research_paper_tests.config_schema import (
    SESSION_EVOLUTION_MODE_ENV,
    get_test_scenario,
)
from research_paper_tests.stats_utils import run_monte_carlo


def test_session_evolution_override_updates_default_sim_config(monkeypatch, n_seeds):
    monkeypatch.setenv(SESSION_EVOLUTION_MODE_ENV, "with")

    def runner():
        config = get_test_scenario("agent_memory").sim_config()
        return config.enable_evolution is True

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    assert all(results)


def test_session_evolution_override_updates_default_run_profile(monkeypatch, n_seeds):
    monkeypatch.setenv(SESSION_EVOLUTION_MODE_ENV, "without")

    def runner():
        run = get_test_scenario("agent_memory").run_profile()
        return run.enable_evolution is False

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    assert all(results)


def test_session_evolution_override_respects_locked_baseline_pairs(monkeypatch, n_seeds):
    monkeypatch.setenv(SESSION_EVOLUTION_MODE_ENV, "with")

    def runner():
        baseline = get_test_scenario("wealth_gini_baseline").sim_config()
        evolved = get_test_scenario("wealth_gini_evolved").sim_config()
        return baseline.enable_evolution is False and evolved.enable_evolution is True

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    assert all(results)
