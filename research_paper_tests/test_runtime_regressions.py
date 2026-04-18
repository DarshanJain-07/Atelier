from collections import OrderedDict

import pytest
import torch

from main import (
    SOCIETY_CACHE,
    SOCIETY_CACHE_LOCK,
    aggregate_social_state,
    prepare_society_sync,
)
from research_paper_tests.config_schema import (
    get_test_scenario,
    prepare_scenario_society,
    set_emotions,
    zero_emotions,
)

SMALL_POPULATION_SETTINGS = get_test_scenario("runtime_small_populations").settings()


@pytest.fixture
def isolated_society_cache():
    with SOCIETY_CACHE_LOCK:
        snapshot = OrderedDict(SOCIETY_CACHE.items())
        SOCIETY_CACHE.clear()

    try:
        yield
    finally:
        with SOCIETY_CACHE_LOCK:
            SOCIETY_CACHE.clear()
            SOCIETY_CACHE.update(snapshot)


def test_prepare_society_cache_isolates_topology_variants(tmp_path, isolated_society_cache):
    topo_scenario = get_test_scenario("runtime_profile_topology")
    topo_run = topo_scenario.run_profile()
    topo_settings = topo_scenario.settings()
    flat_run = topo_run.model_copy(
        update={"use_network_topology": topo_settings["flat_use_network_topology"]},
    )

    _, _, _, _, _, _, adjacency_topo, warnings_topo = prepare_society_sync(
        topo_run, str(tmp_path / "topology"),
    )
    _, _, _, _, _, _, adjacency_flat, warnings_flat = prepare_society_sync(
        flat_run, str(tmp_path / "flat"),
    )

    assert adjacency_topo is not None
    assert adjacency_flat is None
    assert warnings_topo == []
    assert warnings_flat == []

    with SOCIETY_CACHE_LOCK:
        assert len(SOCIETY_CACHE) == 2


def test_prepare_society_cache_isolates_evolution_variants(tmp_path, isolated_society_cache):
    evolved_scenario = get_test_scenario("runtime_profile_evolution")
    evolved_run = evolved_scenario.run_profile()
    evolved_settings = evolved_scenario.settings()
    baseline_run = evolved_run.model_copy(
        update={"enable_evolution": evolved_settings["baseline_enable_evolution"]},
    )
    expected_classes = {
        "Underclass",
        "Working Class",
        "Middle Class",
        "Upper Middle",
        "Elite",
    }

    _, evolved_meta, _, _, _, _, _, _ = prepare_society_sync(
        evolved_run, str(tmp_path / "evolved"),
    )
    _, baseline_meta, _, _, _, _, _, _ = prepare_society_sync(
        baseline_run, str(tmp_path / "baseline"),
    )

    assert set(baseline_meta["Class"].unique()).issubset(expected_classes)
    assert set(evolved_meta["Class"].unique()).issubset(expected_classes)
    assert not baseline_meta["Class"].eq("Agent").any()
    assert not evolved_meta["Class"].eq("Agent").any()
    assert not baseline_meta.equals(evolved_meta)
    assert not baseline_meta["Raw_Wealth"].equals(evolved_meta["Raw_Wealth"])

    with SOCIETY_CACHE_LOCK:
        assert len(SOCIETY_CACHE) == 2


def test_prepare_society_returns_fresh_request_memory(tmp_path, isolated_society_cache):
    memory_scenario = get_test_scenario("runtime_profile_memory")
    run = memory_scenario.run_profile()
    memory_settings = memory_scenario.settings()

    _, _, exposures, _, _, memory_first, _, _ = prepare_society_sync(
        run, str(tmp_path / "first"),
    )
    memory_first[0, 0] = memory_settings["memory_marker"]

    _, _, _, _, _, memory_second, _, _ = prepare_society_sync(
        run, str(tmp_path / "second"),
    )

    assert memory_first.data_ptr() != memory_second.data_ptr()
    assert memory_second.shape == exposures.shape
    assert memory_second[0, 0].item() == pytest.approx(
        memory_settings["empty_memory_value"],
    )


@pytest.mark.parametrize("num_agents", SMALL_POPULATION_SETTINGS["population_sizes"])
def test_generated_topology_handles_small_populations(tmp_path, num_agents):
    scenario = get_test_scenario("runtime_small_populations")
    config = scenario.sim_config(num_agents=num_agents)
    society = prepare_scenario_society(
        "runtime_small_populations",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name=f"society_{num_agents}",
        num_agents=num_agents,
    )

    adjacency = society.adjacency_matrix
    assert adjacency is not None
    assert adjacency.shape == (num_agents, num_agents)

    row_sums = torch.sparse.sum(adjacency, dim=1).to_dense()
    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=SMALL_POPULATION_SETTINGS["row_sum_tolerance"],
    )


def test_acting_ratio_uses_scoped_population_size():
    scenario = get_test_scenario("runtime_acting_ratio")
    config = scenario.sim_config()
    settings = scenario.settings()
    emotions = zero_emotions(settings["acting_population"])
    set_emotions(emotions, settings["acting_emotion"])
    influence = torch.ones(settings["acting_population"])
    engagement = torch.ones(settings["acting_population"])

    state = aggregate_social_state(
        config,
        emotions,
        influence,
        engagement_scores=engagement,
    )

    assert state["population_size"] == settings["acting_population"]
    assert state["acting_count"] == settings["acting_population"]
    assert state["acting_ratio"] == pytest.approx(settings["expected_acting_ratio"])
